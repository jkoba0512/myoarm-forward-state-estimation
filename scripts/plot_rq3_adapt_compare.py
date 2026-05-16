"""Plot F_adapt_compare: closed-loop min-tip across adaptation rules.

For each cell, compares 6 estimators on min-tip error:
  1. K=0 (forward-only baseline)
  2. K=1 (sensor-only baseline)
  3. default reliability (β₀=0, β₁=0.5)
  4. global SPSA β (full-grid trained)
  5. per-cell β training-time last-20-mean
  6. per-cell β deployed (10 fresh episodes)

Two-row figure: d=0 cells top, d=18 cells bottom.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FIG_DIR = Path("figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

CELLS = [("none", 0), ("high", 0), ("xhigh", 0),
         ("none", 18), ("high", 18), ("xhigh", 18)]


def get_kvals(closed_loop_csv: Path, estimators: list[str]) -> dict:
    """Read mean min-tip per (estimator, cell) from closed-loop metrics."""
    df = pd.read_csv(closed_loop_csv)
    df = df[df["estimator"].isin(estimators)]
    agg = (
        df.groupby(["estimator", "noise_condition", "delay_steps"])[
            "min_tip_error"
        ].mean().reset_index()
    )
    return {
        (row["estimator"], row["noise_condition"], int(row["delay_steps"])):
            row["min_tip_error"]
        for _, row in agg.iterrows()
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cl-fullgrid", type=Path,
                   default=Path("runs/closed_loop/2026-05-14T01-18-16Z/metrics.csv"))
    p.add_argument("--per-cell-summary", type=Path,
                   default=Path("runs/per_cell_beta_diagnostic/2026-05-15T12-59-01Z/summary.csv"))
    p.add_argument("--per-cell-deployed", type=Path,
                   default=Path("runs/per_cell_beta_diagnostic/2026-05-15T12-59-01Z/deployed_eval.csv"))
    p.add_argument("--out-stem", type=str, default="F_adapt_compare")
    args = p.parse_args()

    cl = get_kvals(args.cl_fullgrid,
                   ["K=0.0", "K=1.0",
                    "reliability_adaptive_v1",
                    "reliability_adaptive_v2_fullgrid"])
    pcs = pd.read_csv(args.per_cell_summary)
    pcd = pd.read_csv(args.per_cell_deployed)

    pcs_map = {(row["noise"], int(row["delay"])): row["last_20_outcome_mean"]
                for _, row in pcs.iterrows()}
    pcd_map = {(row["noise"], int(row["delay"])): row["deployed_min_tip_mean"]
                for _, row in pcd.iterrows()}

    labels = ["$K{=}0$", "$K{=}1$", "default reliab.",
              "global SPSA $\\beta$", "per-cell $\\beta$ training",
              "per-cell $\\beta$ deployed"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c",
              "#9467bd", "#bcbd22", "#d62728"]

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=False)
    for row, delay in enumerate([0, 18]):
        ax = axes[row]
        cells_delay = [(n, d) for (n, d) in CELLS if d == delay]
        x = np.arange(len(cells_delay))
        width = 0.13

        for i, label in enumerate(labels):
            vals = []
            for (n, d) in cells_delay:
                if i == 0:
                    vals.append(cl[("K=0.0", n, d)])
                elif i == 1:
                    vals.append(cl[("K=1.0", n, d)])
                elif i == 2:
                    vals.append(cl[("reliability_adaptive_v1", n, d)])
                elif i == 3:
                    vals.append(cl[("reliability_adaptive_v2_fullgrid", n, d)])
                elif i == 4:
                    vals.append(pcs_map[(n, d)])
                elif i == 5:
                    vals.append(pcd_map[(n, d)])
            ax.bar(x + (i - 2.5) * width, vals, width,
                   label=label, color=colors[i], alpha=0.9,
                   edgecolor="black", linewidth=0.3)

        ax.set_xticks(x)
        ax.set_xticklabels([f"({n}, $d{{=}}{d}$)" for (n, d) in cells_delay],
                          fontsize=10)
        ax.axhline(0.05, color="grey", linestyle=":", linewidth=0.5, alpha=0.5)
        ax.set_ylabel("min-tip error (m)")
        ax.set_title(f"delay $d {{=}} {delay}$ cells", fontsize=11)
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", alpha=0.25)
        if row == 0:
            ax.legend(fontsize=8, ncol=3, loc="upper right", frameon=False)

    fig.suptitle("RQ3: adaptation rules comparison — $d{=}0$ vs $d{=}18$ regimes",
                 fontsize=12)
    fig.tight_layout()
    pdf = FIG_DIR / f"{args.out_stem}.pdf"
    png = FIG_DIR / f"{args.out_stem}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight", dpi=150)
    print(f"wrote {pdf} and {png}")


if __name__ == "__main__":
    main()
