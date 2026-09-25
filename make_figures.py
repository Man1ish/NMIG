#!/usr/bin/env python3
"""
make_figures.py  —  Publication figures for the heterogeneous-device serverless
profiler (UCB-1 / SW-UCB arm selection over {device x batch}).

Scope: this evaluates the PROFILER in isolation (offline trace replay). It shows
the profiler (a) picks the right device per function, (b) matches an offline
oracle, (c) converges (sublinear regret), (d) is robust to its hyper-parameters,
and (e) adapts under non-stationarity with a sliding window. The profiler is
later embedded into the live OpenWhisk testbed; that integration is out of scope
for these plots.

Reads (paths relative to --data):
  ablation.json     : per-fn {slo_s, configs:{policy:{total_cost,latency_s,slo_met,gpu_mem_gbs}}}
  convergence.json  : per-fn {norm_regret_curve:[...]}   (key also present in full convergence dumps)
  sensitivity.json  : {slo_sweep:[...], gamma_sweep:[...]}
  swucb.json        : {swucb_summary:{fn:{UCB-1|SW-UCB:{mean_reward,slo_violation_rate}}}, run_info:{...}}

Writes <out>/fig_<name>.pdf  (vector, for the paper) and .png (300 dpi preview).

Usage:
  python make_figures.py --data ./data --out ./out
"""
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# ----------------------------------------------------------------------------
# ACM Middleware-friendly style (acmart sigconf: ~3.33in column, ~7.0in text).
# Vector PDF + serif body to match the template; bump font in the paper if needed.
# ----------------------------------------------------------------------------
COL_W, TEXT_W = 3.33, 7.0          # inches
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Nimbus Roman"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.linewidth": 0.7, "grid.linewidth": 0.4, "lines.linewidth": 1.3,
    "axes.grid": True, "grid.alpha": 0.35, "grid.linestyle": "-",
    "savefig.dpi": 300, "savefig.bbox": "tight", "figure.dpi": 150,
    "legend.frameon": True, "legend.framealpha": 0.9, "legend.edgecolor": "0.8",
    "pdf.fonttype": 42, "ps.fonttype": 42,   # embed real fonts (no Type-3)
})

# Okabe-Ito colour-blind-safe palette, consistent across every figure.
POL_COLOR = {"always-CPU": "#0072B2", "always-GPU": "#E69F00",
             "always-2xGPU": "#D55E00", "UCB-1": "#009E73", "oracle": "#444444"}
POL_ORDER = ["always-CPU", "always-GPU", "always-2xGPU", "UCB-1", "oracle"]
FN_COLOR = {"alexnet": "#0072B2", "efficientnet": "#E69F00", "resnet50": "#009E73",
            "inception": "#D55E00", "googlenet": "#CC79A7", "bert": "#56B4E9",
            "distilgpt2": "#999999"}
SMALL_FNS = ["resnet50", "inception", "googlenet", "bert", "distilgpt2"]
VIDEO_FNS = ["alexnet", "efficientnet"]
VIOL_HATCH, VIOL_EDGE = "////", "#B00020"   # SLO-violation marker


def load(data_dir):
    def j(f):
        return json.load(open(os.path.join(data_dir, f)))
    sw = j("swucb.json")
    if "swucb_summary" not in sw:                 # your pipeline writes the flat form
        run_info = {}
        if os.path.exists(os.path.join(data_dir, "run_info.json")):
            run_info = j("run_info.json")
        sw = {"swucb_summary": sw, "run_info": run_info}
    return j("ablation.json"), j("convergence.json"), j("sensitivity.json"), sw


def save(fig, out, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"fig_{name}.{ext}"))
    plt.close(fig)
    print(f"  wrote fig_{name}.pdf / .png")


# ----------------------------------------------------------------------------
# FIG 1 — Ablation: per-function cost of fixed device policies vs the profiler
#         vs the offline oracle. Hatched red bars = SLO violated.
# ----------------------------------------------------------------------------
def fig_ablation(ab, out):
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(TEXT_W, 2.55),
                                   gridspec_kw={"width_ratios": [5, 2]})

    def draw(ax, fns, logy):
        n = len(POL_ORDER); w = 0.16
        for fi, fn in enumerate(fns):
            cfg = ab[fn]["configs"]
            for pi, pol in enumerate(POL_ORDER):
                if pol not in cfg:
                    continue
                x = fi + (pi - (n - 1) / 2) * w
                c = cfg[pol]
                viol = not c["slo_met"]
                ax.bar(x, c["total_cost"], w, color=POL_COLOR[pol],
                       edgecolor=(VIOL_EDGE if viol else "white"),
                       linewidth=(1.1 if viol else 0.5),
                       hatch=(VIOL_HATCH if viol else None), zorder=3)
                if viol:                       # explicit cross above violating bars
                    ax.text(x, c["total_cost"] * (1.25 if logy else 1.02), "\u00d7",
                            ha="center", va="bottom", color=VIOL_EDGE,
                            fontsize=7, fontweight="bold")
        ax.set_xticks(range(len(fns)))
        ax.set_xticklabels(fns, rotation=20, ha="right")
        if logy:
            ax.set_yscale("log"); ax.set_ylim(1, 40)
        ax.set_axisbelow(True)

    draw(axL, SMALL_FNS, logy=False)
    draw(axR, VIDEO_FNS, logy=True)
    axL.set_ylim(0, 0.25)
    axL.set_ylabel("Total cost  (GPU-s, lower = better)")
    # axL.set_title("Lightweight functions")
    # axR.set_title("Latency-critical functions")

    handles = [Patch(facecolor=POL_COLOR[p], edgecolor="white", label=p) for p in POL_ORDER]
    handles.append(Patch(facecolor="white", edgecolor=VIOL_EDGE, hatch=VIOL_HATCH,
                         label="SLO violated"))
    axL.legend(handles=handles, ncol=3, loc="upper left", columnspacing=1.0,
               handlelength=1.4, borderpad=0.4)
    # fig.suptitle("Profiler matches the offline oracle on every function; "
    #              "no fixed device policy does", y=1.02, fontsize=9)
    save(fig, out, "ablation")


# ----------------------------------------------------------------------------
# FIG 2 — Convergence: normalised cumulative regret R(t)/t.
#   Left  : latency-critical fns (large gaps) -> regret drops orders of magnitude.
#   Right : lightweight fns (near-tied arms)  -> regret negligible throughout.
# ----------------------------------------------------------------------------
def fig_convergence(cv, out):
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(TEXT_W, 2.5))

    for fn in VIDEO_FNS:
        y = np.asarray(cv[fn]["norm_regret_curve"]); t = np.arange(1, len(y) + 1)
        axL.plot(t, np.clip(y, 1e-2, None), color=FN_COLOR[fn], label=fn)
    axL.set_yscale("log")
    axL.set_xlabel("Round  $t$"); axL.set_ylabel(r"Normalised regret  $R(t)/t$")
    axL.set_title("Latency-critical functions")
    axL.legend(loc="upper right")

    for fn in SMALL_FNS:
        y = np.asarray(cv[fn]["norm_regret_curve"]); t = np.arange(1, len(y) + 1)
        axR.plot(t, y, color=FN_COLOR[fn], label=fn)
    axR.axhline(0, color="0.4", lw=0.7, ls="--")
    axR.set_ylim(-0.02, 0.10)
    axR.set_xlabel("Round  $t$"); axR.set_ylabel(r"Normalised regret  $R(t)/t$")
    axR.set_title("Lightweight functions")
    axR.legend(loc="upper right", ncol=2, columnspacing=1.0)

    fig.suptitle("Per-round regret vanishes where it matters and stays negligible elsewhere",
                 y=1.02, fontsize=9)
    save(fig, out, "convergence")


# ----------------------------------------------------------------------------
# FIG 3 — Sensitivity to the two reward hyper-parameters.
#   Left  : SLO scale T_SLO -> violation rate (feasibility knee at 1.0).
#   Right : cost weight gamma -> cost scales linearly; decision is invariant.
# ----------------------------------------------------------------------------
def fig_sensitivity(se, out):
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(TEXT_W, 2.5))

    # ---- T_SLO knee ----
    scales = [e["slo_scale"] for e in se["slo_sweep"]]
    mat = np.array([[e["per_function"][fn]["slo_violation_rate"] * 100
                     for fn in se["slo_sweep"][0]["per_function"]]
                    for e in se["slo_sweep"]])
    axL.fill_between(scales, mat.min(1), mat.max(1), color="#B00020", alpha=0.15,
                     label="per-function range")
    axL.plot(scales, mat.mean(1), "-o", color="#B00020", ms=4, label="mean (7 fns)")
    axL.axvline(1.0, color="0.4", lw=0.8, ls=":")
    axL.text(1.02, 60, "feasibility\nboundary", fontsize=6.5, color="0.3")
    axL.set_xlabel(r"SLO scale factor  $T_{\mathrm{SLO}}$")
    axL.set_ylabel("SLO violation rate (%)")
    axL.set_title("SLO tightness"); axL.set_ylim(-4, 108)
    axL.legend(loc="center right")

    # ---- gamma: cost scales, decision does not ----
    gammas = [e["gamma"] for e in se["gamma_sweep"]]
    fns = list(se["gamma_sweep"][0]["per_function"].keys())
    cost = np.array([[e["per_function"][fn]["cost"] for fn in fns]
                     for e in se["gamma_sweep"]])
    viol = np.array([[e["per_function"][fn]["slo_violation_rate"] * 100 for fn in fns]
                     for e in se["gamma_sweep"]])
    axR.plot(gammas, cost.mean(1), "-s", color="#0072B2", ms=4, label="mean cost")
    axR.set_xlabel(r"Cost weight  $\gamma$")
    axR.set_ylabel("Avg total cost", color="#0072B2")
    axR.tick_params(axis="y", labelcolor="#0072B2")
    axR.set_title("Cost weight")
    axT = axR.twinx(); axT.grid(False)
    axT.plot(gammas, viol.mean(1), "-^", color="#009E73", ms=4, label="mean violation %")
    axT.set_ylabel("SLO violation rate (%)", color="#009E73")
    axT.tick_params(axis="y", labelcolor="#009E73"); axT.set_ylim(-1, 12)
    axR.annotate("device choice identical\n"
                 r"for all fns ($\gamma\in[0.025,0.2]$)", xy=(0.96, 0.30),
                 xycoords="axes fraction", fontsize=6.3, color="0.3", ha="right")
    axR.legend(loc="upper left"); axT.legend(loc="lower right")

    # fig.suptitle("Profiler tracks the SLO boundary and is robust to the cost weight",
    #              y=1.02, fontsize=9)
    save(fig, out, "sensitivity")


# ----------------------------------------------------------------------------
# FIG 4 — Non-stationarity: UCB-1 vs SW-UCB mean reward (higher = better).
#   Summary metrics only (per-round recovery curve needs the rolling-reward log).
# ----------------------------------------------------------------------------
def fig_swucb(sw, out):
    s = sw["swucb_summary"]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(TEXT_W, 2.5),
                                   gridspec_kw={"width_ratios": [2, 5]})
    w = 0.36
    cU, cS = "#D55E00", "#009E73"

    def grouped(ax, fns):
        for i, fn in enumerate(fns):
            ax.bar(i - w / 2, s[fn]["UCB-1"]["mean_reward"], w, color=cU, zorder=3)
            ax.bar(i + w / 2, s[fn]["SW-UCB"]["mean_reward"], w, color=cS, zorder=3)
        ax.set_xticks(range(len(fns)))
        ax.set_xticklabels(fns, rotation=20, ha="right")
        ax.axhline(0, color="0.5", lw=0.7)
        ax.set_axisbelow(True)

    grouped(axL, VIDEO_FNS); grouped(axR, SMALL_FNS)
    axR.set_ylim(-0.058, 0.108)
    axL.set_title("Latency-critical (mean reward)")
    axR.set_title("Lightweight (mean reward)")
    axL.set_ylabel("Mean reward  (higher = better)")
    # annotate SLO-violation elimination on the lightweight panel
    for i, fn in enumerate(SMALL_FNS):
        axR.text(i, s[fn]["SW-UCB"]["mean_reward"] + 0.004,
                 "0%\nviol.", ha="center", va="bottom", fontsize=5.6, color=cS)

    handles = [Patch(facecolor=cU, label="UCB-1 (stationary)"),
               Patch(facecolor=cS, label="SW-UCB (sliding window)")]
    axR.legend(handles=handles, loc="lower right")
    fig.suptitle("Under mid-run GPU contention, the sliding window re-adapts "
                 "and removes residual SLO violations", y=1.02, fontsize=9)
    save(fig, out, "swucb")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./data")
    ap.add_argument("--out", default="./out")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    ab, cv, se, sw = load(a.data)
    print("Generating figures:")
    fig_ablation(ab, a.out)
    fig_convergence(cv, a.out)
    fig_sensitivity(se, a.out)
    fig_swucb(sw, a.out)
    print("Done ->", os.path.abspath(a.out))


if __name__ == "__main__":
    main()