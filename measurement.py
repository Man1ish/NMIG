"""
Measurement layer.

Records (function, arm) -> latency/cost outcomes ONCE on real hardware, with
optional repeats so we get variance. Everything downstream replays from the
saved JSON, so the expensive sweeps and ablation never re-run your models.

Two recorders:

  RealRecorder    - calls your actual measure_latency_and_cost (subprocess).
                    Use this on the OpenWhisk box to produce measurements.json.
  SyntheticRecorder - calibrated stand-in so the pipeline is runnable anywhere
                    (e.g. here) without GPUs. Numbers track the trends in the
                    paper: CPU fast for tiny inputs, GPU wins for video/batch,
                    2xGPU helps only large inputs, mem grows with batch.

The saved record schema, per (fid, device, batch):
    {
      "latency_s":   [list of repeats],
      "cpu_mem_gbs": [...],
      "gpu_time_s":  [...],
      "gpu_mem_gbs": [...]
    }
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from profiler_core import Arm, build_arms_for_function


# ---------------------------------------------------------------------------
# Real recorder: wraps the user's existing measurement function
# ---------------------------------------------------------------------------

class RealRecorder:
    """Adapter around the user's measure_latency_and_cost(arm).

    On the OpenWhisk machine, do::

        from profiler import measure_latency_and_cost, function_cfg
        rec = RealRecorder(measure_latency_and_cost, function_cfg)
        rec.record_all(repeats=5, out_path="measurements.json")

    measure_latency_and_cost must return
        (latency_s, cpu_mem_s, gpu_time_s, gpu_mem_s)
    exactly as in the original code. cpu_mem_s / gpu_mem_s are GB*s.
    """

    def __init__(self, measure_fn, function_cfg: dict):
        self.measure_fn = measure_fn
        self.function_cfg = function_cfg

    def record_all(self, repeats: int = 5, out_path: str = "measurements.json") -> dict:
        records: dict[str, dict] = {}
        for fid, cfg in self.function_cfg.items():
            arms = build_arms_for_function(fid, cfg)
            for arm in arms:
                key = _arm_key(arm)
                lat, cpu, gput, gpum = [], [], [], []
                for _ in range(repeats):
                    l, c, gt, gm = self.measure_fn(arm)
                    lat.append(float(l)); cpu.append(float(c))
                    gput.append(float(gt)); gpum.append(float(gm))
                records.setdefault(fid, {})[key] = {
                    "latency_s": lat, "cpu_mem_gbs": cpu,
                    "gpu_time_s": gput, "gpu_mem_gbs": gpum,
                }
        Path(out_path).write_text(json.dumps(records, indent=2))
        return records
    
    
    def record_all(self, repeats: int = 5, out_path: str = "measurements.json") -> dict:
        import time as _time
        records: dict[str, dict] = {}

        # count total arms up front so we can show "arm i/N"
        total_arms = sum(len(build_arms_for_function(fid, cfg))
                         for fid, cfg in self.function_cfg.items())
        arm_i = 0
        t_start = _time.time()

        for fid, cfg in self.function_cfg.items():
            arms = build_arms_for_function(fid, cfg)
            for arm in arms:
                arm_i += 1
                key = _arm_key(arm)
                print(f"[{arm_i}/{total_arms}] {fid:12s} {key:10s} "
                      f"x{repeats} ...", end="", flush=True)
                lat, cpu, gput, gpum = [], [], [], []
                t_arm = _time.time()
                for _ in range(repeats):
                    l, c, gt, gm = self.measure_fn(arm)
                    lat.append(float(l)); cpu.append(float(c))
                    gput.append(float(gt)); gpum.append(float(gm))
                dt = _time.time() - t_arm
                print(f" done  mean_exec={sum(lat)/len(lat):.3f}s  "
                      f"({dt:.1f}s for {repeats} reps)", flush=True)
                records.setdefault(fid, {})[key] = {
                    "latency_s": lat, "cpu_mem_gbs": cpu,
                    "gpu_time_s": gput, "gpu_mem_gbs": gpum,
                }
                # write incrementally so a crash mid-run keeps finished arms
                Path(out_path).write_text(json.dumps(records, indent=2))

        print(f"\nAll {total_arms} arms recorded in "
              f"{_time.time() - t_start:.0f}s -> {out_path}")
        return records


# ---------------------------------------------------------------------------
# Synthetic recorder: calibrated, runs anywhere
# ---------------------------------------------------------------------------

# Per-function behaviour, calibrated from the paper's Section II + your tables.
# base_cpu_latency: seconds at batch=1 on CPU
# gpu_speedup: >1 means GPU faster than CPU at batch=1
_SYNTH = {
    "alexnet":      dict(base_cpu=26.32, gpu_speedup=1.42, cpu_mem_mb=1024),  # video task
    "efficientnet": dict(base_cpu=30.0,  gpu_speedup=1.7,  cpu_mem_mb=1024),  # video task
    "resnet50":     dict(base_cpu=2.40,  gpu_speedup=1.2,  cpu_mem_mb=1024),
    "inception":    dict(base_cpu=2.30,  gpu_speedup=1.15, cpu_mem_mb=1024),
    "googlenet":    dict(base_cpu=2.10,  gpu_speedup=1.10, cpu_mem_mb=1024),
    "bert":         dict(base_cpu=2.0,   gpu_speedup=2.0,  cpu_mem_mb=1024),
    "distilgpt2":   dict(base_cpu=1.8,   gpu_speedup=1.8,  cpu_mem_mb=1024),
}


class SyntheticRecorder:
    """Calibrated synthetic measurements using the real function_cfg tables."""

    def __init__(self, function_cfg: dict, seed: int = 0):
        self.function_cfg = function_cfg
        self.rng = np.random.default_rng(seed)

    def _one(self, fid: str, cfg: dict, arm: Arm) -> tuple[float, float, float, float]:
        s = _SYNTH[fid]
        base = cfg["batch"]
        # batch multiplier index into the cfg tables
        bmul = arm.batch / base if base else 1
        # nearest index in batch_sequence
        seq = cfg["batch_sequence"]
        idx = min(range(len(seq)), key=lambda i: abs(seq[i] - bmul))

        cpu_mem_mb = cfg["memory_mb"]

        if arm.device == "CPU":
            # CPU scales ~linearly with batch
            lat = s["base_cpu"] * (1.0 + 0.8 * (seq[idx] - 1))
            gpu_time = 0.0
            gpu_mem_mb = 0.0
        elif arm.device == "GPU":
            compute = (s["base_cpu"] / s["gpu_speedup"]) * (1.0 + 0.15 * (seq[idx] - 1))
            lat = compute
            gpu_time = compute
            tbl = cfg["gpu1_mb"]
            gpu_mem_mb = tbl[min(idx, len(tbl) - 1)]
        else:  # 2xGPU
            compute = (s["base_cpu"] / s["gpu_speedup"]) * (1.0 + 0.15 * (seq[idx] - 1))
            # multi-GPU helps only big (video) tasks
            penalty = 0.85 if s["base_cpu"] > 10 else 1.4
            lat = compute * penalty
            gpu_time = lat
            g1 = cfg["2xgpu"]["gpu1_mb"]
            g2 = cfg["2xgpu"]["gpu2_mb"]
            gpu_mem_mb = g1[min(idx, len(g1) - 1)] + g2[min(idx, len(g2) - 1)]

        # multiplicative noise
        lat *= math.exp(self.rng.normal(0.0, 0.08))
        cpu_mem_gbs = (cpu_mem_mb / 1024.0) * lat
        gpu_mem_gbs = (gpu_mem_mb / 1024.0) * gpu_time
        return lat, cpu_mem_gbs, gpu_time, gpu_mem_gbs

    def record_all(self, repeats: int = 5, out_path: str = "measurements.json") -> dict:
        records: dict[str, dict] = {}
        for fid, cfg in self.function_cfg.items():
            if fid not in _SYNTH:
                continue
            arms = build_arms_for_function(fid, cfg)
            for arm in arms:
                key = _arm_key(arm)
                lat, cpu, gput, gpum = [], [], [], []
                for _ in range(repeats):
                    l, c, gt, gm = self._one(fid, cfg, arm)
                    lat.append(l); cpu.append(c); gput.append(gt); gpum.append(gm)
                records.setdefault(fid, {})[key] = {
                    "latency_s": lat, "cpu_mem_gbs": cpu,
                    "gpu_time_s": gput, "gpu_mem_gbs": gpum,
                }
        Path(out_path).write_text(json.dumps(records, indent=2))
        return records


# ---------------------------------------------------------------------------
# Replay: sample a recorded outcome for a given arm
# ---------------------------------------------------------------------------

class Replayer:
    """Serves recorded measurements back to the profiler during sweeps.

    Each call to sample(arm) draws one of the recorded repeats at random,
    so the profiler still sees realistic run-to-run variance without
    launching any real inference.
    """

    def __init__(self, records: dict, seed: int = 0):
        self.records = records
        self.rng = np.random.default_rng(seed)

    @classmethod
    def from_file(cls, path: str, seed: int = 0) -> "Replayer":
        return cls(json.loads(Path(path).read_text()), seed=seed)

    def sample(self, arm: Arm) -> tuple[float, float, float, float]:
        rec = self.records[arm.fid][_arm_key(arm)]
        i = int(self.rng.integers(len(rec["latency_s"])))
        return (
            rec["latency_s"][i], rec["cpu_mem_gbs"][i],
            rec["gpu_time_s"][i], rec["gpu_mem_gbs"][i],
        )

    def mean_metrics(self, arm: Arm) -> dict:
        rec = self.records[arm.fid][_arm_key(arm)]
        return {
            "latency_s": float(np.mean(rec["latency_s"])),
            "latency_std": float(np.std(rec["latency_s"])),
            "cpu_mem_gbs": float(np.mean(rec["cpu_mem_gbs"])),
            "gpu_time_s": float(np.mean(rec["gpu_time_s"])),
            "gpu_mem_gbs": float(np.mean(rec["gpu_mem_gbs"])),
        }

    def arms_for(self, fid: str, function_cfg: dict) -> list[Arm]:
        return build_arms_for_function(fid, function_cfg[fid])


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _arm_key(arm: Arm) -> str:
    return f"{arm.device}|{arm.batch}"


def derive_per_function_slo(records: dict, percentile: float = 95.0,
                            slack: float = 1.10) -> dict[str, float]:
    """Per-function SLO = warm p95 latency of the BEST (lowest-latency) arm,
    times a small slack factor. This is the principled, reproducible rule
    that answers Reviewer 2's "explain the basis for TSLO".

    Using the best arm's p95 means the SLO is achievable by at least one
    configuration, which is what makes SLO attainment a meaningful metric.
    """
    slo: dict[str, float] = {}
    for fid, arms in records.items():
        best_p95 = min(
            float(np.percentile(m["latency_s"], percentile)) for m in arms.values()
        )
        slo[fid] = round(best_p95 * slack, 3)
    slo["_default"] = 2.0
    return slo
