"""Supplement violin: episode-level min-tip distribution for the C1
pair (K=0 vs default reliability) across the focused 6-cell grid.

Codex round-5b/5c Q7 recommendation: show the bimodal "reach /
don't reach" episode distribution that sits behind the C1 mean
comparisons, so reviewers can see that the large standard
deviations are a property of the distribution, not of the
estimator-difference signal.

Layout (per codex):
  rows:    delay = 0  /  delay = 18
  cols:    noise = none / high / xhigh
  hue:     K=0  vs  default reliability
  plot:    violin (full distribution) with the n=200 mean overlaid

Source: runs/closed_loop/<200-ep-run>/metrics.csv .

CLI::

    uv run python scripts/plot_supplement_violin.py \\
        --metrics runs/closed_loop/<run>/metrics.csv \\
        --output figures/F_supplement_violin.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

NOISE_ORDER = ["none", "high", "xhigh"]
DELAY_ORDER = (0, 18)
ESTIMATORS = ["K=0.0", "reliability_adaptive_v1"]
LABELS = ["$K{=}0$", "default reliability"]
COLOURS = ["#1f77b4", "#2ca02c"]


def plot_violin(
    df: pd.DataFrame,
    *,
    output: Path,
    metric: str = "min_tip_error",
    figsize: tuple[float, float] = (8.5, 4.6),
    ylim: tuple[float, float] = (-0.05, 1.5),
) -> None:
    df = df.copy()
    df["delay_steps"] = df["delay_steps"].astype(int)
    fig, axes = plt.subplots(
        len(DELAY_ORDER), len(NOISE_ORDER),
        figsize=figsize, sharey=True, sharex=True,
    )
    for r, delay in enumerate(DELAY_ORDER):
        for c, noise in enumerate(NOISE_ORDER):
            ax = axes[r, c]
            data = []
            means = []
            for est in ESTIMATORS:
                vals = df[
                    (df["estimator"] == est)
                    & (df["noise_condition"] == noise)
                    & (df["delay_steps"] == delay)
                ][metric].to_numpy()
                data.append(vals)
                means.append(vals.mean() if vals.size else np.nan)

            parts = ax.violinplot(
                data, positions=[0, 1], widths=0.7,
                showmedians=False, showmeans=False,
                showextrema=False,
            )
            for body, colour in zip(parts["bodies"], COLOURS):
                body.set_facecolor(colour)
                body.set_alpha(0.55)
                body.set_edgecolor(colour)

            for i, (vals, colour, mean) in enumerate(
                zip(data, COLOURS, means)
            ):
                # Mean marker
                ax.scatter([i], [mean], marker="D", s=22,
                           color="black", zorder=5)
                # Median + IQR whiskers (more robust than min/max)
                q1, q2, q3 = np.quantile(vals, [0.25, 0.5, 0.75])
                ax.plot([i, i], [q1, q3], color="black",
                        lw=1.6, zorder=4)
                ax.scatter([i], [q2], marker="_", s=120,
                           color="black", zorder=5)

            ax.set_xticks([0, 1])
            ax.set_xticklabels(LABELS, fontsize=8)
            ax.set_ylim(*ylim)
            ax.grid(axis="y", alpha=0.2)
            if r == 0:
                ax.set_title(f"noise = {noise}", fontsize=9)
            if c == 0:
                ax.set_ylabel(f"delay $d{{=}}{delay}$\n"
                              f"min-tip (m)", fontsize=9)

    fig.suptitle("Episode-level $\\min$-tip distributions: $K{=}0$ "
                 "vs default reliability ($n{=}200$ per cell)",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    png = output.with_suffix(".png")
    fig.savefig(png, bbox_inches="tight", dpi=150)
    print(f"  wrote {output} and {png}")
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metrics", type=Path, required=True)
    p.add_argument("--output", type=Path,
                   default=Path("figures/F_supplement_violin.pdf"))
    args = p.parse_args()
    df = pd.read_csv(args.metrics)
    plot_violin(df, output=args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
