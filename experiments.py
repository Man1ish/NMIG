"""
Experiments, each mapped to a specific reviewer comment.

  experiment_convergence   -> R4.3 (how often does exploration pick sub-optimal
                              arms / violate SLOs; variance evidence)
  experiment_ablation      -> R2 + R4 (isolate profiler contribution vs
                              fixed-arm baselines and oracle)
  experiment_swucb         -> R3.4 (context-free UCB-1 vs non-stationary SW-UCB)
  experiment_slo_gamma     -> R2 + R4.3 (TSLO and gamma sensitivity)

All experiments REPLAY recorded measurements; none launch real inference.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from profiler_core import (
    Arm,
    CostModel,
    ExperimentType,
    FixedArmProfiler,
    SlidingWindowUCBProfiler,
    UCB1Profiler,
    build_arms_for_function,
    compute_cost,
    compute_reward,
    cost_model_for_regime,
)
from measurement import Replayer, _arm_key


# ---------------------------------------------------------------------------
# Oracle: the genuinely best arm per function, from mean recorded reward
# ---------------------------------------------------------------------------

def oracle_best_arm(fid: str, arms: list[Arm], replayer: Replayer, cm: CostModel) -> tuple[Arm, dict]:
    """Best arm by mean reward over recorded repeats (noise averaged out)."""
    scores = {}
    for a in arms:
        m = replayer.mean_metrics(a)
        scores[a] = compute_reward(
            m["latency_s"], m["cpu_mem_gbs"], m["gpu_time_s"], m["gpu_mem_gbs"], cm, fid
        )
    best = max(scores, key=scores.get)
    return best, {str(a): round(v, 4) for a, v in scores.items()}


# ---------------------------------------------------------------------------
# R4.3 : convergence + sub-optimal exploration rate
# ---------------------------------------------------------------------------

def experiment_convergence(function_cfg, replayer, cm, rounds=200, seed=0,
                           tol=0.05, window=20):
    """For each function, run UCB-1 for `rounds` and track convergence.

    Convergence signals reported:
      1. near-optimal selection rate, as a TRAILING-WINDOW fraction (last
         `window` rounds), so early exploration is not counted forever.
         An arm is near-optimal if its mean reward is within `tol` of the
         best arm's mean reward, measured on an ABSOLUTE scale (|best|),
         so several near-tied arms all count as near-optimal -- which is
         the correct behavior for short functions whose arms barely differ.
      2. normalized cumulative regret R(t)/t against the oracle mean reward.
      3. realized cost gap vs oracle: cumulative (cost_ucb - cost_oracle)
         / cost_oracle. This matches the actual claim -- near-oracle COST --
         and stays near zero for short functions even when arm identity churns.
    """
    out = {}
    for fid, cfg in function_cfg.items():
        if fid not in replayer.records:
            continue
        arms = build_arms_for_function(fid, cfg)
        oracle, arm_scores = oracle_best_arm(fid, arms, replayer, cm)

        arm_mean_reward = {}
        arm_mean_cost = {}
        for a in arms:
            m = replayer.mean_metrics(a)
            arm_mean_reward[a] = compute_reward(
                m["latency_s"], m["cpu_mem_gbs"], m["gpu_time_s"], m["gpu_mem_gbs"], cm, fid
            )
            arm_mean_cost[a] = compute_cost(
                m["cpu_mem_gbs"], m["gpu_time_s"], m["gpu_mem_gbs"], cm
            )
        best_reward = arm_mean_reward[oracle]
        oracle_cost = arm_mean_cost[oracle]

        # Near-optimal on an ABSOLUTE scale: within tol * |best_reward|.
        # Near-tied arms (short functions) therefore ALL count as near-optimal.
        denom = abs(best_reward) if abs(best_reward) > 1e-9 else 1.0
        near_set = {a for a in arms
                    if (best_reward - arm_mean_reward[a]) / denom <= tol}

        prof = UCB1Profiler(arms, seed=seed)
        T = cm.slo_for(fid)

        picked_oracle, picked_near = [], []
        slo_violations = 0
        rewards, realized_cost = [], []
        for t in range(rounds):
            a = prof.select_arm()
            lat, cpu, gput, gpum = replayer.sample(a)
            r = compute_reward(lat, cpu, gput, gpum, cm, fid)
            prof.update(a, r)
            picked_oracle.append(1 if a == oracle else 0)
            picked_near.append(1 if a in near_set else 0)
            realized_cost.append(compute_cost(cpu, gput, gpum, cm))
            if lat > T:
                slo_violations += 1
            rewards.append(r)

        idx = np.arange(1, rounds + 1)
        cum_oracle = np.cumsum(picked_oracle) / idx
        cum_near = np.cumsum(picked_near) / idx
        windowed_near = _trailing_rate(picked_near, window)
        norm_regret = best_reward - (np.cumsum(rewards) / idx)

        # realized cost gap vs an oracle that always pays oracle_cost
        cum_cost_ucb = np.cumsum(realized_cost)
        cum_cost_oracle = oracle_cost * idx
        cost_gap = (cum_cost_ucb - cum_cost_oracle) / np.maximum(cum_cost_oracle, 1e-9)

        out[fid] = {
            "oracle_arm": str(oracle),
            "selected_best_arm": str(prof.best_arm()),
            "converged_correct": bool(prof.best_arm() == oracle),
            "near_optimal_arms": [str(a) for a in near_set],
            "n_near_optimal": len(near_set),
            "final_fraction_oracle": float(cum_oracle[-1]),
            "final_fraction_near": float(cum_near[-1]),
            "final_windowed_near": float(windowed_near[-1]),
            "rounds_to_90pct_near": _first_above(windowed_near, 0.90),
            "final_cost_gap": float(cost_gap[-1]),
            "slo_violation_rate": slo_violations / rounds,
            "convergence_curve": cum_oracle.tolist(),
            "near_optimal_curve": cum_near.tolist(),          # cumulative, kept
            "windowed_near_curve": windowed_near.tolist(),    # NEW: use this one
            "norm_regret_curve": norm_regret.tolist(),
            "cost_gap_curve": cost_gap.tolist(),              # NEW: matches claim
            "arm_scores": arm_scores,
            "pull_counts": {str(a): prof.counts[a] for a in arms},
        }
    return out


def _trailing_rate(flags, window):
    f = np.asarray(flags, dtype=float)
    out = np.empty(len(f))
    for t in range(len(f)):
        lo = max(0, t - window + 1)
        out[t] = f[lo:t + 1].mean()
    return out


def _first_above(curve, thresh):
    idx = np.argmax(curve >= thresh) if np.any(curve >= thresh) else -1
    return int(idx + 1) if idx >= 0 else None


# ---------------------------------------------------------------------------
# R2 + R4 : ablation (profiler vs fixed-arm baselines vs oracle)
# ---------------------------------------------------------------------------

def experiment_ablation(function_cfg, replayer, cm, rounds=200, seed=0):
    """Compare, per function, the cost/latency/SLO of:
      - UCB-1 profiler's chosen config
      - always-CPU, always-GPU, always-2xGPU (where available) at base batch
      - oracle best arm

    This isolates the profiler's contribution: if UCB-1's pick matches the
    oracle and beats the fixed baselines, the profiler demonstrably helps.
    Directly answers the most-requested comment (ablation of contributions).
    """
    out = {}
    for fid, cfg in function_cfg.items():
        if fid not in replayer.records:
            continue
        arms = build_arms_for_function(fid, cfg)
        oracle, _ = oracle_best_arm(fid, arms, replayer, cm)

        # UCB-1 learned choice
        prof = UCB1Profiler(arms, seed=seed)
        for _ in range(rounds):
            a = prof.select_arm()
            prof.update(a, compute_reward(*replayer.sample(a), cm, fid))
        ucb_arm = prof.best_arm()

        # Fixed baselines at base batch
        base = cfg["batch"]
        candidates = {
            "always-CPU":  Arm(fid, "CPU", base),
            "always-GPU":  Arm(fid, "GPU", base),
        }
        if "2xGPU" in cfg["devices"]:
            candidates["always-2xGPU"] = Arm(fid, "2xGPU", base)
        candidates["UCB-1"] = ucb_arm
        candidates["oracle"] = oracle

        rows = {}
        for label, a in candidates.items():
            if a not in arms:
                continue
            m = replayer.mean_metrics(a)
            cost = compute_cost(m["cpu_mem_gbs"], m["gpu_time_s"], m["gpu_mem_gbs"], cm)
            rows[label] = {
                "arm": str(a),
                "latency_s": round(m["latency_s"], 3),
                "latency_std": round(m["latency_std"], 3),
                "total_cost": round(cost, 4),
                "gpu_mem_gbs": round(m["gpu_mem_gbs"], 4),
                "slo_met": bool(m["latency_s"] <= cm.slo_for(fid)),
            }
        out[fid] = {"slo_s": cm.slo_for(fid), "configs": rows,
                    "ucb_matches_oracle": bool(ucb_arm == oracle)}
    return out


# ---------------------------------------------------------------------------
# R3.4 : UCB-1 vs sliding-window UCB
# ---------------------------------------------------------------------------

def experiment_swucb(function_cfg, replayer, cm, rounds=300, window=50,
                     drift_at=None, seed=0):
    """Head-to-head UCB-1 vs SW-UCB per function.

    If `drift_at` is set, after that round the replayer is told GPU arms got
    slower (contention), so the previously-best arm degrades. A stationary
    UCB-1 adapts slowly; SW-UCB should react faster. This is the
    non-stationary scenario Reviewer 3 implies.

    Returns mean reward and SLO violations for each profiler, plus the
    full reward trace so we can plot the adaptation.
    """
    out = {}
    for fid, cfg in function_cfg.items():
        if fid not in replayer.records:
            continue
        arms = build_arms_for_function(fid, cfg)
        results = {}
        for Prof in (UCB1Profiler, SlidingWindowUCBProfiler):
            prof = (Prof(arms, window=window, seed=seed)
                    if Prof is SlidingWindowUCBProfiler else Prof(arms, seed=seed))
            T = cm.slo_for(fid)
            rewards, viols = [], 0
            for t in range(rounds):
                a = prof.select_arm()
                lat, cpu, gput, gpum = replayer.sample(a)
                # inject drift: GPU arms slow down 2x after drift_at
                if drift_at is not None and t >= drift_at and a.device in ("GPU", "2xGPU"):
                    lat *= 2.0
                    gput *= 2.0
                r = compute_reward(lat, cpu, gput, gpum, cm, fid)
                prof.update(a, r)
                rewards.append(r)
                if lat > T:
                    viols += 1
            results[prof.name] = {
                "mean_reward": float(np.mean(rewards)),
                "slo_violation_rate": viols / rounds,
                "reward_trace": rewards,
            }
        out[fid] = results
    return out


# ---------------------------------------------------------------------------
# R2 + R4.3 : TSLO and gamma sensitivity
# ---------------------------------------------------------------------------

def experiment_slo_gamma(function_cfg, replayer, base_slo, rounds=150, seed=0):
    """Sweep TSLO scale and gamma, report how the profiler's chosen config
    and resulting cost/SLO change. Converts 'ad hoc setting' into a
    characterized operating point (R2), and shows exploration's SLO impact
    across regimes (R4.3).
    """
    slo_scales = [0.5, 0.75, 1.0, 1.5, 2.0]
    gammas = [0.0, 0.025, 0.05, 0.10, 0.20]

    def run_once(cm):
        per_fn = {}
        for fid, cfg in function_cfg.items():
            if fid not in replayer.records:
                continue
            arms = build_arms_for_function(fid, cfg)
            prof = UCB1Profiler(arms, seed=seed)
            T = cm.slo_for(fid)
            viols = 0
            for _ in range(rounds):
                a = prof.select_arm()
                lat, cpu, gput, gpum = replayer.sample(a)
                prof.update(a, compute_reward(lat, cpu, gput, gpum, cm, fid))
                if lat > T:
                    viols += 1
            chosen = prof.best_arm()
            m = replayer.mean_metrics(chosen)
            per_fn[fid] = {
                "chosen": str(chosen),
                "cost": round(compute_cost(m["cpu_mem_gbs"], m["gpu_time_s"], m["gpu_mem_gbs"], cm), 4),
                "slo_violation_rate": viols / rounds,
            }
        return per_fn

    slo_sweep = []
    for sc in slo_scales:
        scaled = {k: (v * sc if k != "_default" else v) for k, v in base_slo.items()}
        cm = CostModel(gamma_cpu=0.05, gamma_time=0.05, gamma_mem=0.025, T_slo=scaled)
        slo_sweep.append({"slo_scale": sc, "per_function": run_once(cm)})

    gamma_sweep = []
    for g in gammas:
        cm = CostModel(gamma_cpu=g, gamma_time=g, gamma_mem=g / 2, T_slo=base_slo)
        gamma_sweep.append({"gamma": g, "per_function": run_once(cm)})

    return {"slo_sweep": slo_sweep, "gamma_sweep": gamma_sweep}
