"""Generate the 3 figures introduced by the reliability-adaptive reframe:

* F_reliability_default.pdf : closed-loop min-tip per (noise, delay) cell for
  K=0, K=1, and the default-reliability observer (β0=0, β1=0.5) under H=8 +
  joint-PD. Reads runs/closed_loop/2026-05-14T01-18-16Z/metrics.csv.

* F_spsa_single.pdf : single-cell SPSA convergence (top) and final per-field
  β bar chart (bottom). Reads runs/reliability_adaptive_v2/2026-05-13T12-24-25Z/.

* F_spsa_fullgrid.pdf : multi-cell SPSA convergence (top) and per-cell
  min-tip bar chart at iteration 100 for K=0, default-reliability, and the
  multi-cell-trained β (bottom). Reads
  runs/reliability_adaptive_v2/2026-05-13T23-40-35Z/ for SPSA history and
  runs/closed_loop/2026-05-14T01-18-16Z/metrics.csv for per-cell bars.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FIG_DIR = Path("figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

CL_REACHFIXED = "runs/closed_loop/2026-05-14T01-18-16Z/metrics.csv"
CL_REACHFIXED_200EP = "runs/closed_loop/2026-05-19T01-46-59Z/metrics.csv"
CL_C2_SINGLE_200EP = "runs/closed_loop/2026-05-21T06-05-22Z/metrics.csv"
SPSA_SINGLE = "runs/reliability_adaptive_v2/2026-05-13T12-24-25Z"
SPSA_FULLGRID = "runs/reliability_adaptive_v2/2026-05-13T23-40-35Z"

NOISE_ORDER = ["none", "high", "xhigh"]
DELAY_ORDER = [0, 18]
FIELDS = ["qpos", "qvel", "act", "tip_pos", "reach_err"]


def _cell_label(noise: str, delay: int) -> str:
    return f"{noise}\nd={delay}"


def _save(fig: plt.Figure, name: str) -> None:
    out_pdf = FIG_DIR / f"{name}.pdf"
    out_png = FIG_DIR / f"{name}.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=150)
    print(f"  wrote {out_pdf} and {out_png}")


def fig_reliability_default() -> None:
    print("F_reliability_default")
    df = pd.read_csv(CL_REACHFIXED)
    keep = {"K=0.0", "K=1.0", "reliability_adaptive_v1"}
    df = df[df["estimator"].isin(keep)].copy()
    df["delay_steps"] = df["delay_steps"].astype(int)

    # Aggregate to mean/std per (estimator, noise, delay)
    agg = (
        df.groupby(["estimator", "noise_condition", "delay_steps"])["min_tip_error"]
        .agg(["mean", "std"])
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    cells = [(n, d) for d in DELAY_ORDER for n in NOISE_ORDER]
    x = np.arange(len(cells))
    width = 0.27

    est_order = ["K=0.0", "K=1.0", "reliability_adaptive_v1"]
    labels = ["$K{=}0$", "$K{=}1$", "default reliability"]
    colours = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    for i, est in enumerate(est_order):
        means = []
        stds = []
        for (n, d) in cells:
            row = agg[(agg["estimator"] == est) & (agg["noise_condition"] == n) & (agg["delay_steps"] == d)]
            means.append(row["mean"].iloc[0])
            stds.append(row["std"].iloc[0])
        ax.bar(x + (i - 1) * width, means, width, yerr=stds,
               label=labels[i], color=colours[i], capsize=2, alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels([_cell_label(n, d) for (n, d) in cells], fontsize=8)
    ax.set_ylabel("closed-loop $\\min$-tip error (m)")
    ax.axhline(0.05, color="grey", linestyle="--", linewidth=0.6, alpha=0.6)
    ax.set_title("Default within-trial reliability vs $K{=}0$ / $K{=}1$ ($H{=}8$, joint-PD)",
                 fontsize=10)
    ax.legend(fontsize=8, loc="upper left", frameon=False)
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.25)
    _save(fig, "F_reliability_default")
    plt.close(fig)


def _load_spsa_history(run_dir: str):
    h = json.load(open(f"{run_dir}/history.json"))
    fb = json.load(open(f"{run_dir}/final_beta.json"))
    iters = np.array([e["iter"] for e in h])
    outcome_mean = np.array([e["outcome_mean"] for e in h])
    return iters, outcome_mean, fb


def fig_spsa_single() -> None:
    print("F_spsa_single")
    iters, outcome, fb = _load_spsa_history(SPSA_SINGLE)
    # K=0 baseline at (none, d=18): use the n=200 closed-loop sweep so the
    # dashed reference matches the C1 reporting standard.
    cl = pd.read_csv(CL_REACHFIXED_200EP)
    k0_baseline = cl[(cl["estimator"] == "K=0.0") &
                     (cl["noise_condition"] == "none") &
                     (cl["delay_steps"] == 18)]["min_tip_error"].mean()
    # Deployed evaluation of the final C2 β at n=200 on the same cell.
    c2 = pd.read_csv(CL_C2_SINGLE_200EP)
    c2_deployed = c2[(c2["estimator"] == "reliability_adaptive_v2_c2_single") &
                     (c2["noise_condition"] == "none") &
                     (c2["delay_steps"] == 18)]["min_tip_error"].mean()

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(5.6, 4.6),
        gridspec_kw={"height_ratios": [1.0, 0.85], "hspace": 0.42},
    )

    # Top: outcome trajectory + smoothed running mean
    ax_top.plot(iters, outcome, color="#1f77b4", linewidth=0.7, alpha=0.45,
                label="outcome per iter")
    win = 10
    smooth = np.convolve(outcome, np.ones(win) / win, mode="valid")
    ax_top.plot(iters[win - 1:], smooth, color="#1f77b4", linewidth=1.8,
                label=f"{win}-iter running mean")
    ax_top.axhline(k0_baseline, color="#ff7f0e", linestyle="--", linewidth=1.0,
                   label=f"$K{{=}}0$ baseline ({k0_baseline:.2f} m, $n{{=}}200$)")
    ax_top.axhline(c2_deployed, color="#2ca02c", linestyle=":", linewidth=1.0,
                   label=f"deployed C2 $\\beta$ ({c2_deployed:.2f} m, $n{{=}}200$)")
    ax_top.set_xlabel("SPSA iteration")
    ax_top.set_ylabel("per-iter $\\min$-tip (m)")
    ax_top.set_title("Single-cell SPSA: $(\\sigma{=}\\mathrm{none},\\,d{=}18)$, $H{=}8$, $S{=}10$",
                     fontsize=10)
    ax_top.legend(fontsize=8, loc="upper right", frameon=False)
    ax_top.grid(alpha=0.25)
    ax_top.set_ylim(min(0.35, outcome.min() * 0.95), max(0.65, outcome.max() * 1.05))

    # Bottom: final β bar chart (β0 and β1 side by side per field)
    b0 = np.array([fb["beta0"][f] for f in FIELDS])
    b1 = np.array([fb["beta1"][f] for f in FIELDS])
    x = np.arange(len(FIELDS))
    ax_bot.bar(x - 0.2, b0, 0.4, color="#9467bd", label="$\\beta_{0,f}$")
    ax_bot.bar(x + 0.2, b1, 0.4, color="#d62728", label="$\\beta_{1,f}$")
    ax_bot.axhline(0, color="black", linewidth=0.6)
    ax_bot.set_xticks(x)
    ax_bot.set_xticklabels(FIELDS, fontsize=9)
    ax_bot.set_ylabel("final $\\beta$ value")
    ax_bot.legend(fontsize=8, loc="lower left", frameon=False)
    ax_bot.grid(axis="y", alpha=0.25)
    ax_bot.set_title("Final per-field $\\beta$ at iter 100", fontsize=10)

    _save(fig, "F_spsa_single")
    plt.close(fig)


def fig_spsa_fullgrid() -> None:
    print("F_spsa_fullgrid")
    iters, outcome, _ = _load_spsa_history(SPSA_FULLGRID)

    # Per-cell bars at the end of training (eval data)
    df = pd.read_csv(CL_REACHFIXED)
    keep = {"K=0.0", "reliability_adaptive_v1", "reliability_adaptive_v2_fullgrid"}
    df = df[df["estimator"].isin(keep)].copy()
    df["delay_steps"] = df["delay_steps"].astype(int)
    agg = (
        df.groupby(["estimator", "noise_condition", "delay_steps"])["min_tip_error"]
        .agg(["mean", "std"])
        .reset_index()
    )

    # Multi-cell-mean K=0 baseline for the top panel reference line
    cells = [(n, d) for d in DELAY_ORDER for n in NOISE_ORDER]
    k0_means = []
    for n, d in cells:
        m = agg[(agg["estimator"] == "K=0.0") &
                (agg["noise_condition"] == n) &
                (agg["delay_steps"] == d)]["mean"].iloc[0]
        k0_means.append(m)
    k0_grid_mean = float(np.mean(k0_means))

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(6.4, 5.0),
        gridspec_kw={"height_ratios": [0.9, 1.0], "hspace": 0.42},
    )

    # Top: outcome trajectory
    ax_top.plot(iters, outcome, color="#1f77b4", linewidth=0.7, alpha=0.45,
                label="outcome per iter")
    win = 10
    smooth = np.convolve(outcome, np.ones(win) / win, mode="valid")
    ax_top.plot(iters[win - 1:], smooth, color="#1f77b4", linewidth=1.8,
                label=f"{win}-iter running mean")
    ax_top.axhline(k0_grid_mean, color="#ff7f0e", linestyle="--", linewidth=1.0,
                   label=f"$K{{=}}0$ grid mean ({k0_grid_mean:.2f} m)")
    ax_top.set_xlabel("SPSA iteration")
    ax_top.set_ylabel("6-cell mean $\\min$-tip (m)")
    ax_top.set_title("Multi-cell SPSA: 6-cell uniform sampling, $H{=}8$, $S{=}12$",
                     fontsize=10)
    ax_top.legend(fontsize=8, loc="upper right", frameon=False)
    ax_top.grid(alpha=0.25)

    # Bottom: per-cell bars
    x = np.arange(len(cells))
    width = 0.27
    est_order = ["K=0.0", "reliability_adaptive_v1", "reliability_adaptive_v2_fullgrid"]
    labels = ["$K{=}0$", "default reliability", "full-grid SPSA $\\beta$"]
    colours = ["#1f77b4", "#2ca02c", "#9467bd"]
    for i, est in enumerate(est_order):
        means = []
        stds = []
        for (n, d) in cells:
            row = agg[(agg["estimator"] == est) &
                      (agg["noise_condition"] == n) &
                      (agg["delay_steps"] == d)]
            means.append(row["mean"].iloc[0])
            stds.append(row["std"].iloc[0])
        ax_bot.bar(x + (i - 1) * width, means, width, yerr=stds,
                   label=labels[i], color=colours[i], capsize=2, alpha=0.9)
    ax_bot.set_xticks(x)
    ax_bot.set_xticklabels([_cell_label(n, d) for (n, d) in cells], fontsize=8)
    ax_bot.set_ylabel("closed-loop $\\min$-tip error (m)")
    ax_bot.set_title("Per-cell $\\min$-tip at iter 100", fontsize=10)
    ax_bot.legend(fontsize=8, loc="upper left", frameon=False)
    ax_bot.axhline(0.05, color="grey", linestyle="--", linewidth=0.6, alpha=0.6)
    ax_bot.grid(axis="y", alpha=0.25)
    ax_bot.set_ylim(0, 1.0)

    _save(fig, "F_spsa_fullgrid")
    plt.close(fig)


def main() -> None:
    fig_reliability_default()
    fig_spsa_single()
    fig_spsa_fullgrid()


if __name__ == "__main__":
    main()
