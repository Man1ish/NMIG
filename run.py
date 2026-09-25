"""
Main runner. Produces all reviewer-targeted results and figures.

On your OpenWhisk machine, replace the SyntheticRecorder block with:

    from profiler import measure_latency_and_cost, function_cfg
    from measurement import RealRecorder
    RealRecorder(measure_latency_and_cost, function_cfg).record_all(
        repeats=5, out_path="measurements.json")

then run this file. Here we use the synthetic recorder so the whole
pipeline runs without GPUs.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from profiler_core import CostModel, ExperimentType, cost_model_for_regime
from measurement import Replayer, SyntheticRecorder, derive_per_function_slo
from experiments import (
    experiment_ablation,
    experiment_convergence,
    experiment_slo_gamma,
    experiment_swucb,
)


# --- the user's real function_cfg (alexnet = video task, intended) ----------
function_cfg = {
    "alexnet": {
        "memory_mb": 1024, "batch": 15, "batching": True,
        "gpu1_mb": [588, 608, 634, 668, 720],
        "2xgpu": {"gpu1_mb": [950, 968, 976, 986], "gpu2_mb": [670, 690, 706, 706]},
        "batch_sequence": [1, 2, 3, 4],
        "filename": "alexnet_video.py",
        "filepath": "python_runtime/core/mlruntime/alexnet/alexnet_video.py",
        "devices": ["CPU", "GPU", "2xGPU"],
    },
    "efficientnet": {
        "memory_mb": 1024, "batch": 15, "batching": False,
        "gpu1_mb": [1020, 1434, 1856, 2280],
        "2xgpu": {"gpu1_mb": [954, 1150, 1384, 1584], "gpu2_mb": [888, 1102, 1306, 1528]},
        "batch_sequence": [1, 2, 3, 4],
        "filename": "efficientnet_video.py",
        "filepath": "python_runtime/core/mlruntime/efficientnet/efficientnet_video.py",
        "devices": ["CPU", "GPU", "2xGPU"],
    },
    "resnet50": {
        "memory_mb": 1024, "batch": 1, "batching": False, "gpu1_mb": [528],
        "2xgpu": {"gpu1_mb": [598], "gpu2_mb": [600]}, "batch_sequence": [1],
        "filename": "resnet50.py", "filepath": "...", "devices": ["CPU", "GPU", "2xGPU"],
    },
    "inception": {
        "memory_mb": 1024, "batch": 1, "batching": False, "gpu1_mb": [536],
        "2xgpu": {"gpu1_mb": [630], "gpu2_mb": [630]}, "batch_sequence": [1],
        "filename": "inception.py", "filepath": "...", "devices": ["CPU", "GPU", "2xGPU"],
    },
    "googlenet": {
        "memory_mb": 1024, "batch": 1, "batching": False, "gpu1_mb": [390],
        "2xgpu": {"gpu1_mb": [490], "gpu2_mb": [490]}, "batch_sequence": [1],
        "filename": "googlenet.py", "filepath": "...", "devices": ["CPU", "GPU", "2xGPU"],
    },
    "bert": {
        "memory_mb": 1024, "batch": 1, "batching": False, "gpu1_mb": [778],
        "2xgpu": {"gpu1_mb": [778], "gpu2_mb": [778]}, "batch_sequence": [1],
        "filename": "bert.py", "filepath": "...", "devices": ["CPU", "GPU"],
    },
    "distilgpt2": {
        "memory_mb": 1024, "batch": 1, "batching": False, "gpu1_mb": [668],
        "2xgpu": {"gpu1_mb": [0], "gpu2_mb": [0]}, "batch_sequence": [1],
        "filename": "distilgpt2.py", "filepath": "...", "devices": ["CPU", "GPU"],
    },
}

# OUT and FIG are set per-run by setup_run_dirs(). Defaults keep imports working.
RUNS_ROOT = Path("runs")
OUT = Path("results")
FIG = Path("figures")


def setup_run_dirs(run_name: str | None = None) -> str:
    """Create runs/<run_name>/{results,figures} and point OUT/FIG at them.

    A timestamp is always appended (e.g. balanced_20240528-153012), so
    even a reused name never overwrites a prior run. Refreshes runs/latest.
    Returns the resolved run name.
    """
    global OUT, FIG
    ts = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    # Always append a timestamp so a reused name never overwrites a prior run.
    run_name = f"{run_name}_{ts}" if run_name else ts
    run_dir = RUNS_ROOT / run_name
    OUT = run_dir / "results"
    FIG = run_dir / "figures"
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    # Refresh a convenience pointer to the most recent run.
    latest = RUNS_ROOT / "latest"
    try:
        if latest.is_symlink() or latest.exists():
            if latest.is_symlink():
                latest.unlink()
            elif latest.is_dir():
                shutil.rmtree(latest)
            else:
                latest.unlink()
        latest.symlink_to(run_name)  # relative link inside runs/
    except OSError:
        # Filesystems without symlink support: write a text pointer instead.
        (RUNS_ROOT / "latest.txt").write_text(run_name)
    return run_name


def main(run_name: str | None = None, meas_path: str = "measurements.json"):
    name = setup_run_dirs(run_name)
    print(f"Run: {name}  ->  {OUT.parent}")
    # 1) Record measurements (synthetic here; RealRecorder on your box)
    print("Recording measurements (synthetic)...")
    rec = SyntheticRecorder(function_cfg, seed=0).record_all(repeats=8, out_path=str(OUT / "measurements.json"))
    replayer = Replayer(rec, seed=1)

    # 2) Derive per-function SLO from measured warm p95 (R2 justification)
    slo = derive_per_function_slo(rec, percentile=95.0, slack=1.10)
    (OUT / "slo.json").write_text(json.dumps(slo, indent=2))
    (OUT / "run_info.json").write_text(json.dumps({
        "run_name": name,
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "measurements_source": "synthetic",
        "slo": slo,
    }, indent=2))
    print("Per-function SLO (warm p95 x1.10):")
    for k, v in slo.items():
        print(f"  {k:14s} {v}")

    cm = CostModel(gamma_cpu=0.05, gamma_time=0.05, gamma_mem=0.025, beta=10.0, T_slo=slo)

    # 3) Convergence + sub-optimal exploration (R4.3)
    print("\n[R4.3] convergence / exploration cost ...")
    conv = experiment_convergence(function_cfg, replayer, cm, rounds=200, seed=0)
    (OUT / "convergence.json").write_text(json.dumps(conv, indent=2))
    for fid, d in conv.items():
        print(f"  {fid:14s} oracle={d['oracle_arm']:>18s} picked={d['selected_best_arm']:>18s} "
              f"near_opt_final={d['final_fraction_near']:.2f} "
              f"SLOviol={d['slo_violation_rate']:.2%} "
              f"to90%(near)={d['rounds_to_90pct_near']} n_near={d['n_near_optimal']}")

    # 4) Ablation (R2 + R4)
    print("\n[R2/R4] ablation vs fixed baselines + oracle ...")
    abl = experiment_ablation(function_cfg, replayer, cm, rounds=200, seed=0)
    (OUT / "ablation.json").write_text(json.dumps(abl, indent=2))
    for fid, d in abl.items():
        print(f"  {fid} (SLO {d['slo_s']}s) ucb==oracle: {d['ucb_matches_oracle']}")
        for label, row in d["configs"].items():
            print(f"     {label:14s} {row['arm']:>18s} lat={row['latency_s']:>7.2f} "
                  f"cost={row['total_cost']:>8.3f} slo_met={row['slo_met']}")

    # 5) UCB-1 vs SW-UCB with drift (R3.4)
    print("\n[R3.4] UCB-1 vs SW-UCB under mid-run GPU contention ...")
    sw = experiment_swucb(function_cfg, replayer, cm, rounds=300, window=50, drift_at=150, seed=0)
    (OUT / "swucb.json").write_text(json.dumps(
        {f: {p: {k: v for k, v in r.items() if k != "reward_trace"}
             for p, r in d.items()} for f, d in sw.items()}, indent=2))
    for fid, d in sw.items():
        u = d["UCB-1"]; s = d["SW-UCB"]
        print(f"  {fid:14s} UCB-1 reward={u['mean_reward']:>7.3f} viol={u['slo_violation_rate']:.2%} | "
              f"SW-UCB reward={s['mean_reward']:>7.3f} viol={s['slo_violation_rate']:.2%}")

    # 6) TSLO + gamma sensitivity (R2 + R4.3)
    print("\n[R2/R4.3] TSLO + gamma sensitivity ...")
    sens = experiment_slo_gamma(function_cfg, replayer, slo, rounds=150, seed=0)
    (OUT / "sensitivity.json").write_text(json.dumps(sens, indent=2))
    print("  saved sensitivity.json")

    # 7) Figures
    print("\nGenerating figures ...")
    fig_convergence(conv)
    fig_ablation(abl, cm)
    fig_swucb(sw)
    fig_sensitivity(sens)
    print(f"Done. results/ and figures/ written.")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _rolling(y, k=9):
    """Centered moving average that preserves length without edge droop.
    Uses progressively smaller windows at the boundaries (pandas-style)
    instead of zero-padding, so the first/last points are not pulled down.
    """
    import numpy as _np
    y = _np.asarray(y, dtype=float)
    n = len(y)
    if k <= 1 or n < 3:
        return y
    half = k // 2
    out = _np.empty(n)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        out[i] = y[lo:hi].mean()
    return out


def fig_convergence(conv):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

    # Left: near-optimal selection fraction (within 5% of best arm reward)
    for fid, d in conv.items():
        y = _rolling(d["near_optimal_curve"], k=9)
        axes[0].plot(np.arange(1, len(y) + 1), y, label=fid)
    axes[0].axhline(0.9, ls="--", c="gray", lw=0.8)
    axes[0].axvline(20, ls=":", c="red", lw=1.0, label="old 20-round cutoff")
    axes[0].set_xlabel("Round"); axes[0].set_ylabel("Fraction selecting a near-optimal arm")
    axes[0].set_title("Near-optimal selection (within 5% of best)")
    axes[0].legend(fontsize=8, ncol=2); axes[0].grid(alpha=0.3); axes[0].set_ylim(0, 1.02)

    # Right: normalized cumulative regret R(t)/t -> 0 means converged
    for fid, d in conv.items():
        y = _rolling(d["norm_regret_curve"], k=9)
        axes[1].plot(np.arange(1, len(y) + 1), y, label=fid)
    axes[1].axhline(0.0, ls="--", c="gray", lw=0.8)
    axes[1].set_xlabel("Round"); axes[1].set_ylabel("Normalized regret R(t)/t")
    axes[1].set_title("Regret converges toward zero")
    axes[1].legend(fontsize=8, ncol=2); axes[1].grid(alpha=0.3)

    fig.suptitle("Profiler convergence per function", y=1.02)
    fig.tight_layout(); fig.savefig(FIG / "fig_convergence.pdf"); fig.savefig(FIG / "fig_convergence.png", dpi=150)
    plt.close(fig)


def fig_ablation(abl, cm):
    fids = list(abl.keys())
    labels = ["always-CPU", "always-GPU", "always-2xGPU", "UCB-1", "oracle"]
    colors = {"always-CPU": "#1f77b4", "always-GPU": "#9467bd",
              "always-2xGPU": "#8c564b", "UCB-1": "#2ca02c", "oracle": "#d62728"}
    fig, ax = plt.subplots(figsize=(11, 4.2))
    x = np.arange(len(fids)); w = 0.16
    for i, label in enumerate(labels):
        costs = [abl[f]["configs"].get(label, {}).get("total_cost", 0) for f in fids]
        ax.bar(x + (i - 2) * w, costs, w, label=label, color=colors[label])
    ax.set_xticks(x); ax.set_xticklabels(fids, rotation=20)
    ax.set_ylabel("Total cost (lower better)")
    ax.set_title("Ablation: profiler choice vs fixed baselines vs oracle")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(FIG / "fig_ablation.pdf"); fig.savefig(FIG / "fig_ablation.png", dpi=150)
    plt.close(fig)


def fig_swucb(sw):
    fids = [f for f in sw if f in ("alexnet", "efficientnet")]  # batching fns most interesting
    if not fids:
        fids = list(sw.keys())[:2]
    fig, axes = plt.subplots(1, len(fids), figsize=(5.5 * len(fids), 3.6), squeeze=False)
    for ax, fid in zip(axes[0], fids):
        for pname, c in (("UCB-1", "#d62728"), ("SW-UCB", "#2ca02c")):
            tr = np.array(sw[fid][pname]["reward_trace"])
            # smooth with rolling mean
            k = 10
            sm = np.convolve(tr, np.ones(k) / k, mode="valid")
            ax.plot(sm, label=pname, color=c)
        ax.axvline(150, ls=":", c="black", lw=1.0, label="GPU contention onset")
        ax.set_title(fid); ax.set_xlabel("Round"); ax.set_ylabel("Reward (rolling)")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle("UCB-1 vs SW-UCB under mid-run non-stationarity", y=1.02)
    fig.tight_layout(); fig.savefig(FIG / "fig_swucb.pdf"); fig.savefig(FIG / "fig_swucb.png", dpi=150)
    plt.close(fig)


def fig_sensitivity(sens):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    # TSLO sweep: avg SLO violation across functions
    scales = [r["slo_scale"] for r in sens["slo_sweep"]]
    viol = [np.mean([v["slo_violation_rate"] for v in r["per_function"].values()])
            for r in sens["slo_sweep"]]
    axes[0].plot(scales, np.array(viol) * 100, "o-", color="#d62728")
    axes[0].set_xlabel("SLO scale factor"); axes[0].set_ylabel("Avg SLO violation (%)")
    axes[0].set_title("Sensitivity to TSLO"); axes[0].grid(alpha=0.3)

    gammas = [r["gamma"] for r in sens["gamma_sweep"]]
    cost = [np.mean([v["cost"] for v in r["per_function"].values()])
            for r in sens["gamma_sweep"]]
    axes[1].plot(gammas, cost, "s-", color="#2ca02c")
    axes[1].set_xlabel("gamma"); axes[1].set_ylabel("Avg total cost")
    axes[1].set_title("Sensitivity to gamma"); axes[1].grid(alpha=0.3)

    fig.suptitle("Reward-parameter sensitivity", y=1.02)
    fig.tight_layout(); fig.savefig(FIG / "fig_sensitivity.pdf"); fig.savefig(FIG / "fig_sensitivity.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="NMIG profiler experiments")
    ap.add_argument("--name", default=None,
                    help="run name (folder under runs/). Default: timestamp.")
    args = ap.parse_args()
    main(run_name=args.name)
