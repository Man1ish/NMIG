"""
Main runner. Produces all reviewer-targeted results and figures.

Reads measurements.json (produced by record_real.py) and builds every
experiment and figure from it. Figures are styled for ACM Middleware
acmart sigconf (two-column, 7in text width, ~9pt body).

CAMERA-READY CHANGE (profiler figure consolidation)
---------------------------------------------------
Old Figures 10, 11 and 12 were three separate full-width figure* blocks
holding five panels between them. That costs ~14-15 column-inches plus
three captions. This script emits TWO candidate replacements, into
sibling folders, and leaves the existing figures/ untouched:

  figures/profiler_grid/   FIVE panels, two rows, nothing dropped.
                           ~9 column-inches, one caption. Recommended.

  figures/profiler_row/    THREE panels, one row. Saves the most space but
                           the line-plot panels sit at 1.95in, well under
                           the 3.33in column width. Kept for comparison.

Compile both, look at them, delete the folder you don't use.

Every panel is exported at its FINAL physical width, so \\includegraphics
does no scaling and fonts stay consistent. Do not resize these PDFs in
LaTeX beyond width=\\linewidth on a subfigure of the matching fraction.
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
from matplotlib.patches import Patch
import numpy as np

from profiler_core import CostModel, ExperimentType, cost_model_for_regime
from measurement import Replayer, derive_per_function_slo
from experiments import (
    experiment_ablation,
    experiment_convergence,
    experiment_slo_gamma,
    experiment_swucb,
)


# ---------------------------------------------------------------------------
# Middleware acmart sigconf style. Two-column layout: column = 3.33in,
# text = 7.0in. ACM minimum 7pt; 9pt body stays readable after double-column
# reflow. Applied via rc_context inside each fig_* so it does not leak.
# ---------------------------------------------------------------------------
COL_W, TEXT_W = 3.33, 7.0
MIDDLEWARE_RC = {
    "font.family": "serif",
    "font.serif": ["Linux Libertine O", "DejaVu Serif", "Times New Roman"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 9,
    "axes.titlesize": 9.5, "axes.labelsize": 9,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "legend.fontsize": 8,
    "axes.linewidth": 0.8, "grid.linewidth": 0.4, "lines.linewidth": 1.4,
    "axes.grid": True, "grid.alpha": 0.35, "grid.linestyle": "-",
    "savefig.dpi": 300, "savefig.bbox": "tight", "figure.dpi": 150,
    "legend.frameon": True, "legend.framealpha": 0.92, "legend.edgecolor": "0.8",
    "pdf.fonttype": 42, "ps.fonttype": 42,
}

# ---------------------------------------------------------------------------
# GRID layout (five panels, two rows) - the recommended one.
#
#   row 1:  convergence | cost gap | drift (bert)      0.32\textwidth each
#   row 2:  ablation                | drift (distilgpt2)  0.66 | 0.30
#
# The ablation panel gets 4.60in because it carries 7 functions x 5 policies
# plus a broken axis. The four line plots get 2.24in, which is under the
# 3.33in column width but comfortable for single-series-per-function curves.
# ---------------------------------------------------------------------------
G_LINE_W = 2.24     # 0.32 * 7.0
G_ABL_W = 4.60      # 0.66 * 7.0
G_SMALL_W = 2.10    # 0.30 * 7.0
G_H = 2.10

# ---------------------------------------------------------------------------
# ROW layout (three panels, one row) - the compact alternative.
# ---------------------------------------------------------------------------
R_LINE_W = 1.95     # 0.27 * 7.0
R_ABL_W = 3.05      # 0.43 * 7.0
R_H = 2.05

# Type scales for the two layouts. Axis text stays at or above the ACM 7pt
# floor in both; only the legends drop below, which is a judgement call. If
# a camera-ready check flags it, raise legend.fontsize to 7 and move the
# legend out with _legend_below().
GRID_RC = {
    **MIDDLEWARE_RC,
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 6.5, "lines.linewidth": 1.2,
}
ROW_RC = {
    **MIDDLEWARE_RC,
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 6.2, "lines.linewidth": 1.1,
}

# Okabe-Ito colour-blind-safe palette, consistent across every figure.
POL_COLOR = {"always-CPU": "#0072B2", "always-GPU": "#E69F00",
             "always-2xGPU": "#D55E00", "UCB-1": "#009E73", "oracle": "#444444"}
POL_ORDER = ["always-CPU", "always-GPU", "always-2xGPU", "UCB-1", "oracle"]
FN_COLOR = {"alexnet": "#0072B2", "efficientnet": "#E69F00", "resnet50": "#009E73",
            "inception": "#D55E00", "googlenet": "#CC79A7", "bert": "#56B4E9",
            "distilgpt2": "#999999"}
SMALL_FNS = ["resnet50", "inception", "googlenet", "bert", "distilgpt2"]
VIDEO_FNS = ["alexnet", "efficientnet"]
VIOL_HATCH, VIOL_EDGE = "////", "#B00020"

# distilgpt2 is excluded from the convergence panel because its two arms
# (CPU, GPU) have nearly equal reward, so the bandit explores both
# indefinitely. The ~60% plateau is correct behavior but reads as a failure
# on this metric. Explained in the caption.
CONV_EXCLUDE = {"distilgpt2"}


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

GRID_DIRNAME = "profiler_grid"
ROW_DIRNAME = "profiler_row"


def grid_dir() -> Path:
    d = FIG / GRID_DIRNAME
    d.mkdir(parents=True, exist_ok=True)
    return d


def row_dir() -> Path:
    d = FIG / ROW_DIRNAME
    d.mkdir(parents=True, exist_ok=True)
    return d


def setup_run_dirs(run_name: str | None = None) -> str:
    """Create runs/<run_name>/{results,figures} and point OUT/FIG at them.

    A timestamp is always appended (e.g. balanced_20240528-153012), so even a
    reused name never overwrites a prior run. Refreshes runs/latest.
    """
    global OUT, FIG
    ts = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"{run_name}_{ts}" if run_name else ts
    run_dir = RUNS_ROOT / run_name
    OUT = run_dir / "results"
    FIG = run_dir / "figures"
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    (FIG / GRID_DIRNAME).mkdir(parents=True, exist_ok=True)
    (FIG / ROW_DIRNAME).mkdir(parents=True, exist_ok=True)

    latest = RUNS_ROOT / "latest"
    try:
        if latest.is_symlink() or latest.exists():
            if latest.is_symlink():
                latest.unlink()
            elif latest.is_dir():
                shutil.rmtree(latest)
            else:
                latest.unlink()
        latest.symlink_to(run_name)
    except OSError:
        (RUNS_ROOT / "latest.txt").write_text(run_name)
    return run_name


def main(run_name: str | None = None, meas_path: str = "measurements.json"):
    name = setup_run_dirs(run_name)
    print(f"Run: {name}  ->  {OUT.parent}")

    # 1) Load the REAL measurements you recorded with record_real.py
    import json as _json
    print(f"Loading real measurements from {meas_path} ...")
    rec = _json.loads(open(meas_path).read())
    replayer = Replayer(rec, seed=1)

    # 2) Derive per-function SLO from measured warm p95 (R2 justification)
    slo = derive_per_function_slo(rec, percentile=95.0, slack=1.10)
    (OUT / "slo.json").write_text(json.dumps(slo, indent=2))
    (OUT / "run_info.json").write_text(json.dumps({
        "run_name": name,
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "measurements_source": meas_path,
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
    drift_fids = emit_grid_layout(conv, abl, sw)
    emit_row_layout(conv, abl, sw)
    fig_sensitivity(sens)

    print(f"\nFive-panel grid  -> {grid_dir()}/")
    print(f"Three-panel row  -> {row_dir()}/")
    print(f"Drift panels use: {', '.join(drift_fids)}"
          "   (name these in the subcaptions)")


# ---------------------------------------------------------------------------
# Shared panel drawing. Each helper draws onto an axes it is handed, so the
# grid and row layouts share one implementation and cannot drift apart.
# ---------------------------------------------------------------------------

def _rolling(y, k=9):
    """Centered moving average that preserves length without edge droop."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    if k <= 1 or n < 3:
        return y
    half = k // 2
    out = np.empty(n)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        out[i] = y[lo:hi].mean()
    return out


def _legend_below(ax, ncol=3, y=-0.30, **kw):
    """Put a legend under the axes instead of inside it.

    Use this if a reviewer objects to the sub-7pt in-axes legend, or if a
    panel's legend covers data. It grows the saved PDF downward, so apply it
    to every panel in a row at once or the row will not align on its
    baseline.
    """
    return ax.legend(loc="upper center", bbox_to_anchor=(0.5, y),
                     ncol=ncol, frameon=False, columnspacing=1.0,
                     handlelength=1.3, handletextpad=0.5, **kw)


def _compact_legend(ax, **kw):
    """In-axes legend with the padding squeezed out."""
    opts = dict(columnspacing=0.8, handlelength=1.2, handletextpad=0.4,
                borderpad=0.3, labelspacing=0.25)
    opts.update(kw)
    return ax.legend(**opts)


def _draw_convergence(ax, conv, legend_ncol=2):
    """Near-optimal selection rate per function."""
    for fid, d in conv.items():
        if fid not in FN_COLOR or fid in CONV_EXCLUDE:
            continue
        y = d.get("windowed_near_curve") or d.get("near_optimal_curve")
        if y is None:
            continue
        ax.plot(np.arange(1, len(y) + 1), y, color=FN_COLOR[fid], label=fid)
    ax.axhline(0.9, ls="--", c="gray", lw=0.8)
    ax.axvline(20, ls=":", c=VIOL_EDGE, lw=1.0)
    ax.set_xlabel("Round  $t$")
    ax.set_ylabel("Near-optimal selection rate")
    ax.set_ylim(0, 1.04)
    # Curves finish top-left, so lower right stays clear.
    _compact_legend(ax, loc="lower right", ncol=legend_ncol)


def _draw_costgap(ax, conv, legend_ncol=2):
    """Realised cost gap against the oracle, percent."""
    for fid, d in conv.items():
        if fid not in FN_COLOR:
            continue
        y = d.get("cost_gap_curve")
        if y is None:
            continue
        ys = _rolling(np.asarray(y) * 100, k=9)
        ax.plot(np.arange(1, len(ys) + 1), ys, color=FN_COLOR[fid], label=fid)
    ax.axhline(0.0, ls="--", c="gray", lw=0.8)
    ax.set_xlabel("Round  $t$")
    ax.set_ylabel("Realised cost gap vs oracle (%)")
    _compact_legend(ax, loc="upper right", ncol=legend_ncol)


def _viol_curve(sw, fid, pname):
    """Cumulative SLO violations inferred from the reward trace.

    Rounds where reward < -0.1 are dominated by the SLO penalty: the cost
    term alone is O(0.01-0.1) but the penalty is beta * (L - T_SLO),
    typically O(1-10). If the experiment ever records a per-round violation
    flag directly, read that instead.
    """
    tr = np.asarray(sw[fid][pname]["reward_trace"], dtype=float)
    return np.cumsum((tr < -0.1).astype(int))


def _draw_drift(ax, sw, fid, show_ylabel=True):
    """UCB-1 vs SW-UCB, cumulative SLO violations.

    Cumulative violations rather than rolling reward: reward sits near zero
    most rounds with occasional spikes, so the two policies look identical
    even when their steady-state behaviour differs. The cumulative curve
    shows the divergence directly - stationary UCB-1 keeps climbing after
    the drift, SW-UCB flattens.
    """
    for pname, c in (("UCB-1", "#D55E00"), ("SW-UCB", "#009E73")):
        y = _viol_curve(sw, fid, pname)
        ax.plot(np.arange(1, len(y) + 1), y, color=c, label=pname)
    ax.axvline(150, ls=":", c="black", lw=1.0)
    ax.set_xlabel("Round  $t$")
    if show_ylabel:
        ax.set_ylabel("Cumulative SLO violations")
    # No in-panel title: the function name goes in the subcaption, and a
    # title would make this panel taller than its neighbours.
    _compact_legend(ax, loc="upper left")


def _drift_order(sw):
    """Functions sorted by how much SW-UCB helps, best first."""
    def gap(fid):
        return (sw[fid]["UCB-1"]["slo_violation_rate"]
                - sw[fid]["SW-UCB"]["slo_violation_rate"])
    return sorted(sw.keys(), key=gap, reverse=True)


def _draw_ablation(abl, axL, axR, tick_rot=30, legend_ncol=2, mark_fs=6.5,
                   headroom=1.45):
    """Per-function total cost: fixed policies, UCB-1, oracle.

    axL takes the small functions on a linear axis, axR the video functions
    on a log axis. Pass axR=None to put everything on one axes.
    SLO-violating bars are hatched and marked with a cross.
    """
    small = [f for f in abl if f in SMALL_FNS]
    video = [f for f in abl if f in VIDEO_FNS]
    panels = [(axL, small, False)]
    if axR is not None and video:
        panels.append((axR, video, True))

    n = len(POL_ORDER); w = 0.16
    for ax, fns, logy in panels:
        for fi, fn in enumerate(fns):
            cfg = abl[fn]["configs"]
            for pi, pol in enumerate(POL_ORDER):
                if pol not in cfg:
                    continue
                x = fi + (pi - (n - 1) / 2) * w
                c = cfg[pol]
                viol = not c.get("slo_met", True)
                ax.bar(x, c["total_cost"], w, color=POL_COLOR[pol],
                       edgecolor=(VIOL_EDGE if viol else "white"),
                       linewidth=(1.0 if viol else 0.4),
                       hatch=(VIOL_HATCH if viol else None), zorder=3)
                if viol:
                    ax.text(x, c["total_cost"] * (1.28 if logy else 1.04),
                            "\u00d7", ha="center", va="bottom",
                            color=VIOL_EDGE, fontsize=mark_fs, fontweight="bold")
        ax.set_xticks(range(len(fns)))
        ax.set_xticklabels(fns, rotation=tick_rot, ha="right")
        if logy:
            ax.set_yscale("log"); ax.set_ylim(1, 40)
        ax.set_axisbelow(True)

    axL.set_ylabel("Total cost")
    max_cost = max(abl[f]["configs"].get(p, {"total_cost": 0})["total_cost"]
                   for f in panels[0][1] for p in POL_ORDER)
    # Headroom so the legend clears the tallest bar.
    axL.set_ylim(0, max(0.25, max_cost * headroom))

    handles = [Patch(facecolor=POL_COLOR[p], edgecolor="white", label=p)
               for p in POL_ORDER]
    handles.append(Patch(facecolor="white", edgecolor=VIOL_EDGE,
                         hatch=VIOL_HATCH, label="SLO violated"))
    _compact_legend(axL, handles=handles, ncol=legend_ncol, loc="upper left")


# ---------------------------------------------------------------------------
# Layout A: five panels, two rows. Nothing dropped.
# ---------------------------------------------------------------------------

def emit_grid_layout(conv, abl, sw):
    """Write the five grid panels. Returns the drift functions plotted."""
    out = grid_dir()
    order = _drift_order(sw)
    top_fid = order[0]                       # largest SW-UCB advantage
    second_fid = order[1] if len(order) > 1 else None

    with plt.rc_context(GRID_RC):
        # (a) convergence
        fig, ax = plt.subplots(figsize=(G_LINE_W, G_H))
        _draw_convergence(ax, conv)
        fig.tight_layout(pad=0.3)
        fig.savefig(out / "fig_profiler_convergence.pdf")
        fig.savefig(out / "fig_profiler_convergence.png", dpi=150)
        plt.close(fig)

        # (b) cost gap vs oracle
        fig, ax = plt.subplots(figsize=(G_LINE_W, G_H))
        _draw_costgap(ax, conv)
        fig.tight_layout(pad=0.3)
        fig.savefig(out / "fig_profiler_costgap.pdf")
        fig.savefig(out / "fig_profiler_costgap.png", dpi=150)
        plt.close(fig)

        # (c) drift, strongest case
        fig, ax = plt.subplots(figsize=(G_LINE_W, G_H))
        _draw_drift(ax, sw, top_fid)
        fig.tight_layout(pad=0.3)
        fig.savefig(out / f"fig_profiler_drift_{top_fid}.pdf")
        fig.savefig(out / f"fig_profiler_drift_{top_fid}.png", dpi=150)
        plt.close(fig)

        # (d) ablation, the wide one
        fig, (axL, axR) = plt.subplots(
            1, 2, figsize=(G_ABL_W, G_H), gridspec_kw={"width_ratios": [5, 2]})
        _draw_ablation(abl, axL, axR, tick_rot=25, legend_ncol=3,
                       mark_fs=7, headroom=1.35)
        fig.tight_layout(pad=0.3, w_pad=0.8)
        fig.savefig(out / "fig_profiler_ablation.pdf")
        fig.savefig(out / "fig_profiler_ablation.png", dpi=150)
        plt.close(fig)

        # (e) drift, second case
        if second_fid is not None:
            fig, ax = plt.subplots(figsize=(G_SMALL_W, G_H))
            _draw_drift(ax, sw, second_fid)
            fig.tight_layout(pad=0.3)
            fig.savefig(out / f"fig_profiler_drift_{second_fid}.pdf")
            fig.savefig(out / f"fig_profiler_drift_{second_fid}.png", dpi=150)
            plt.close(fig)

    fids = [f for f in (top_fid, second_fid) if f is not None]
    print(f"  grid  : 5 panels -> {out}")
    for fid in sw:
        u = sw[fid]["UCB-1"]["slo_violation_rate"]
        s = sw[fid]["SW-UCB"]["slo_violation_rate"]
        mark = "  <- plotted" if fid in fids else ""
        print(f"    drift {fid:14s} UCB-1 {u:.2%}  SW-UCB {s:.2%}{mark}")
    return fids


# ---------------------------------------------------------------------------
# Layout B: three panels, one row. Two panels dropped.
# ---------------------------------------------------------------------------

def emit_row_layout(conv, abl, sw):
    """Write the three row panels: convergence, ablation, drift (top case).

    Drops the cost-gap panel (the ablation panel already compares against the
    oracle) and the weaker drift panel. Line panels land at 1.95in, under the
    3.33in column width - that is the price of three across.
    """
    out = row_dir()
    top_fid = _drift_order(sw)[0]

    with plt.rc_context(ROW_RC):
        fig, ax = plt.subplots(figsize=(R_LINE_W, R_H))
        _draw_convergence(ax, conv)
        fig.tight_layout(pad=0.3)
        fig.savefig(out / "fig_profiler_convergence.pdf")
        fig.savefig(out / "fig_profiler_convergence.png", dpi=150)
        plt.close(fig)

        fig, (axL, axR) = plt.subplots(
            1, 2, figsize=(R_ABL_W, R_H), gridspec_kw={"width_ratios": [5, 2]})
        _draw_ablation(abl, axL, axR, tick_rot=35, legend_ncol=2,
                       mark_fs=6.5, headroom=1.45)
        fig.tight_layout(pad=0.3, w_pad=0.6)
        fig.savefig(out / "fig_profiler_ablation.pdf")
        fig.savefig(out / "fig_profiler_ablation.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(R_LINE_W, R_H))
        _draw_drift(ax, sw, top_fid)
        fig.tight_layout(pad=0.3)
        fig.savefig(out / "fig_profiler_drift.pdf")
        fig.savefig(out / "fig_profiler_drift.png", dpi=150)
        plt.close(fig)

    print(f"  row   : 3 panels -> {out}")


# ---------------------------------------------------------------------------
# Unchanged: SLO sensitivity, its own single-column figure in figures/.
# ---------------------------------------------------------------------------

def fig_sensitivity(sens):
    """SLO-scale sensitivity. Single panel: violation rate vs T_SLO scale.

    Left as its own column-width figure. It is profiler evidence too, but a
    sixth panel would push the others below a readable width.

    The gamma sweep is omitted. In the recorded data, scaling gamma over
    [0, 0.2] changes the chosen arm for only one function (distilgpt2, once),
    so the gamma panel plots gamma x fixed-usage against gamma and is not
    informative. The SLO-scale sweep is the substantive sensitivity result:
    it justifies anchoring T_SLO to the measured warm-start p95.
    """
    with plt.rc_context(MIDDLEWARE_RC):
        fig, ax = plt.subplots(1, 1, figsize=(COL_W, 2.4))

        scales = [r["slo_scale"] for r in sens["slo_sweep"]]
        fns_in_sweep = list(sens["slo_sweep"][0]["per_function"].keys())
        mat = np.array([[r["per_function"][fn]["slo_violation_rate"] * 100
                         for fn in fns_in_sweep] for r in sens["slo_sweep"]])

        ax.fill_between(scales, mat.min(1), mat.max(1),
                        color=VIOL_EDGE, alpha=0.15, label="per-function range")
        ax.plot(scales, mat.mean(1), "-o", color=VIOL_EDGE, ms=4.5,
                label=f"mean ({mat.shape[1]} fns)")
        ax.axvline(1.0, color="0.4", lw=0.8, ls=":")
        ax.text(1.05, 35, "feasibility\nboundary", fontsize=7.5, color="0.3",
                va="center")
        ax.set_xlabel(r"SLO scale factor  $T_{\mathrm{SLO}}$")
        ax.set_ylabel("SLO violation rate (%)")
        ax.set_ylim(-4, 108)
        ax.legend(loc="upper right")

        fig.tight_layout(pad=0.4)
        fig.savefig(FIG / "fig_sensitivity.pdf")
        fig.savefig(FIG / "fig_sensitivity.png", dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="NMIG profiler experiments")
    ap.add_argument("--name", default=None,
                    help="run name (folder under runs/). Default: timestamp.")
    ap.add_argument("--meas-path", default="measurements.json",
                    help="path to the measurements.json to replay.")
    args = ap.parse_args()
    main(run_name=args.name, meas_path=args.meas_path)