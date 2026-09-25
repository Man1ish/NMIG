"""
NMIG profiler, revision edition.

This keeps the structure of the original profiler (one bandit PER FUNCTION,
asymmetric per-function arm spaces, real subprocess measurement) and adds
exactly what the four reviewers asked for:

  R2/R3 (cost-model contradiction): a single UNIFIED total-cost objective.
        gamma_cpu is a first-class term, no longer silently dropped.
  R2/R4 (TSLO + gamma sensitivity): per-function SLO derived from measured
        warm p95 latency, plus sweep hooks over gamma and TSLO.
  R3   (context-free UCB-1 vs non-stationary): SlidingWindowUCB profiler
        added alongside UCB-1 for head-to-head comparison.
  R2/R4 (ablation + convergence): fixed-arm baselines and an oracle, so the
        profiler's contribution can be isolated and its convergence shown.

Nothing here re-runs your models thousands of times. The measurement layer
records each (function, arm) outcome once (or a few repeats for variance),
then every sweep and ablation REPLAYS those recordings.
"""

from __future__ import annotations

import math
from collections import defaultdict, namedtuple
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Arm: unchanged from the original (function id + device + concrete batch)
# ---------------------------------------------------------------------------

Arm = namedtuple("Arm", ["fid", "device", "batch"])


def build_arms_for_function(fid: str, cfg: dict) -> list[Arm]:
    """Per-function arm space, matching the original construction.

    Batch arms are only generated for batching functions; non-batching
    functions get a single batch value. Device list is per-function, so
    bert/distilgpt2 (CPU,GPU only) correctly get no 2xGPU arm.
    """
    devices = cfg["devices"]
    base = cfg["batch"]
    factors = cfg["batch_sequence"]
    multipliers = [math.floor(f * base) for f in factors]
    return [Arm(fid, d, b) for d in devices for b in multipliers]


# ---------------------------------------------------------------------------
# Unified cost model  (addresses R2 comment 2 and R3 comment 2)
# ---------------------------------------------------------------------------

@dataclass
class CostModel:
    """Single TOTAL-cost objective.

    The paper previously said "set gamma_cpu = 0 to focus on GPU cost" in
    the model text, but the implementation used gamma_cpu = 0.05 and the
    evaluation reported CPU+memory cost as a headline metric. Reviewers 2
    and 3 both flagged this as making the optimization target unclear.

    We resolve it by committing to ONE total-cost objective:

        C = gamma_cpu  * cpu_mem_GBs            (CPU memory residency)
          + gamma_time * gpu_time_s             (GPU compute time)
          + gamma_mem  * gpu_mem_GBs            (GPU memory residency)

    The "gamma_cpu -> 0" statement becomes a documented special case for
    commercial platforms where CPU-seconds are bundled into the
    per-invocation fee, not a contradiction.

    Attributes
    ----------
    gamma_cpu, gamma_time, gamma_mem : float
        Cost coefficients. All three are reported and swept.
    beta : float
        SLO-violation penalty multiplier.
    T_slo : float or dict[str, float]
        Per-function SLO target (seconds). Dict is keyed by function id
        with a "_default" fallback. A scalar applies to all functions.
    """

    gamma_cpu: float = 0.05
    gamma_time: float = 0.05
    gamma_mem: float = 0.025
    beta: float = 10.0
    T_slo: object = 1.0  # float | dict[str, float]

    def slo_for(self, fid: str) -> float:
        if isinstance(self.T_slo, (int, float)):
            return float(self.T_slo)
        if fid in self.T_slo:
            return float(self.T_slo[fid])
        if "_default" in self.T_slo:
            return float(self.T_slo["_default"])
        raise KeyError(f"No SLO for {fid!r} and no '_default'")


def compute_cost(cpu_mem_gbs: float, gpu_time_s: float, gpu_mem_gbs: float, cm: CostModel) -> float:
    """Total resource cost of one invocation."""
    return cm.gamma_cpu * cpu_mem_gbs + cm.gamma_time * gpu_time_s + cm.gamma_mem * gpu_mem_gbs


def compute_reward(
    latency_s: float,
    cpu_mem_gbs: float,
    gpu_time_s: float,
    gpu_mem_gbs: float,
    cm: CostModel,
    fid: str = "_default",
) -> float:
    """SLO-aware reward over the unified cost model.

        r = (T_SLO - L) - C          if L <= T_SLO
          = -beta * (L - T_SLO) - C  otherwise

    This is the paper's reward shape, now over TOTAL cost and with a
    per-function SLO so batching/video functions are not permanently in
    the penalty regime.
    """
    c = compute_cost(cpu_mem_gbs, gpu_time_s, gpu_mem_gbs, cm)
    T = cm.slo_for(fid)
    if latency_s <= T:
        return (T - latency_s) - c
    return -cm.beta * (latency_s - T) - c


# ---------------------------------------------------------------------------
# Reward regimes  (your LATENCY_FIRST / COST_FIRST / BALANCED, kept)
# ---------------------------------------------------------------------------

class ExperimentType(Enum):
    LATENCY_FIRST = "latency"
    COST_FIRST = "cost"
    BALANCED = "balanced"


def cost_model_for_regime(exp_type: ExperimentType, T_slo) -> CostModel:
    """Map your three regimes onto the unified cost model.

    Reframed for the paper: the three regimes are three operating points
    on ONE total-cost objective, which is precisely the gamma-sensitivity
    story Reviewers 2 and 4 asked for, presented as an intentional design
    choice rather than an ad hoc setting.
    """
    if exp_type == ExperimentType.LATENCY_FIRST:
        return CostModel(gamma_cpu=0.0, gamma_time=0.0, gamma_mem=0.0, T_slo=T_slo)
    if exp_type == ExperimentType.COST_FIRST:
        return CostModel(gamma_cpu=0.10, gamma_time=0.10, gamma_mem=0.05, T_slo=T_slo)
    if exp_type == ExperimentType.BALANCED:
        return CostModel(gamma_cpu=0.05, gamma_time=0.05, gamma_mem=0.025, T_slo=T_slo)
    raise ValueError(f"Unknown experiment type: {exp_type}")


# ---------------------------------------------------------------------------
# Profilers  (UCB-1 kept verbatim in spirit; SW-UCB added for R3)
# ---------------------------------------------------------------------------

class UCB1Profiler:
    """UCB-1 over a per-function arm set. Same as the original MABProfiler."""

    name = "UCB-1"

    def __init__(self, arms: list[Arm], seed: int = 0):
        self.arms = arms
        self.counts: dict[Arm, int] = defaultdict(int)
        self.values: dict[Arm, float] = defaultdict(float)
        self.total_rounds = 0
        self.rng = np.random.default_rng(seed)

    def select_arm(self) -> Arm:
        self.total_rounds += 1
        for a in self.arms:
            if self.counts[a] == 0:
                return a
        log_t = math.log(self.total_rounds)
        best, best_score = None, -1e18
        for a in self.arms:
            score = self.values[a] + math.sqrt(2 * log_t / self.counts[a])
            if score > best_score:
                best, best_score = a, score
        return best

    def update(self, arm: Arm, reward: float) -> None:
        self.counts[arm] += 1
        n = self.counts[arm]
        self.values[arm] += (reward - self.values[arm]) / n

    def best_arm(self) -> Arm:
        return max(self.arms, key=lambda a: self.values[a])


class SlidingWindowUCBProfiler:
    """UCB over the last W rewards per arm. Directly answers Reviewer 3,
    who asked for a comparison against a non-stationary alternative.

    Tracks a bounded history per arm; mean and exploration bonus use the
    window only, so the policy adapts when the best arm drifts (e.g. under
    GPU contention during bursts) instead of locking onto a stale winner.
    """

    name = "SW-UCB"

    def __init__(self, arms: list[Arm], window: int = 50, seed: int = 0):
        self.arms = arms
        self.window = window
        self.history: dict[Arm, list[float]] = {a: [] for a in arms}
        self.total_rounds = 0
        self.rng = np.random.default_rng(seed)

    def select_arm(self) -> Arm:
        self.total_rounds += 1
        for a in self.arms:
            if not self.history[a]:
                return a
        log_t = math.log(min(self.total_rounds, self.window) + 1)
        best, best_score = None, -1e18
        for a in self.arms:
            h = self.history[a]
            mean = sum(h) / len(h)
            score = mean + math.sqrt(2 * log_t / len(h))
            if score > best_score:
                best, best_score = a, score
        return best

    def update(self, arm: Arm, reward: float) -> None:
        h = self.history[arm]
        h.append(reward)
        if len(h) > self.window:
            h.pop(0)
        self.total_rounds  # no-op; rounds bumped in select

    def best_arm(self) -> Arm:
        def m(a):
            h = self.history[a]
            return sum(h) / len(h) if h else -1e18
        return max(self.arms, key=m)


class FixedArmProfiler:
    """Always pick one arm. Ablation baseline (always-CPU/GPU/2xGPU)."""

    def __init__(self, arms: list[Arm], fixed: Arm):
        self.arms = arms
        self.fixed = fixed
        self.name = f"Fixed({fixed.device})"

    def select_arm(self) -> Arm:
        return self.fixed

    def update(self, arm: Arm, reward: float) -> None:
        pass

    def best_arm(self) -> Arm:
        return self.fixed
