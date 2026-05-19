"""Per-delay-split 2-subplot figure with mean +/- 95% bootstrap CI.

Designed for Stage 2 of the 200 ep/cell refresh (codex round-5b Q3):
``F_reliability_default`` and ``F_adapt_compare`` move from a single
bar chart spanning all 6 cells to two side-by-side subplots split by
delay, so that the d=0 (sharp K=0, parameterisation ceiling) and d=18
(achievable, context aggregation ceiling) regimes are visually
separated.

Error bars use mean +/- 95% percentile bootstrap CI; n_bootstrap is
overridable. Significance stars are deliberately *not* drawn -- paired
contrasts go in supplementary tables (see
``compute_paired_bootstrap_ci.py``).

CLI::

    uv run python scripts/plot_per_delay_split.py \\
        --metrics runs/closed_loop/<run>/metrics.csv \\
        --estimators K=0.0 reliability_adaptive_v1 \\
                     reliability_adaptive_v2_fullgrid \\
        --labels '$K{=}0$' 'default reliability' 'global SPSA $\\beta$' \\
        --output figures/F_reliability_default.pdf \\
        --title 'Default reliability vs $K{=}0$ ($H{=}8$, joint-PD)'

The 6-cell mean is *not* drawn on the main panel; it is printed to
stdout so it can be inserted into the figure caption / paper text.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_NOISE_ORDER = ["none", "high", "xhigh"]
DEFAULT_DELAYS = (0, 18)
DEFAULT_COLOURS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2",
]


def bootstrap_mean_ci(
    values: np.ndarray,
    *,
    n_bootstrap: int,
    rng: np.random.Generator,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Return ``(mean, ci_low, ci_high)`` via percentile bootstrap.

    ``values`` is a 1-D array of episode-level samples for one
    (estimator, noise, delay) cell. The CI is independent of the other
    cells (Q3 layout is just visual; Q2 paired contrasts live in
    ``compute_paired_bootstrap_ci.py``).
    """
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    idx = rng.integers(0, values.size, size=(n_bootstrap, values.size))
    boot = values[idx].mean(axis=1)
    lo, hi = np.quantile(boot, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(values.mean()), float(lo), float(hi)


def per_delay_subplot(
    ax: plt.Axes,
    df_delay: pd.DataFrame,
    *,
    estimators: list[str],
    labels: list[str],
    colours: list[str],
    noise_order: list[str],
    metric: str,
    n_bootstrap: int,
    rng: np.random.Generator,
    width: float = 0.27,
) -> None:
    """Draw one delay panel; the caller composes left/right subplots."""
    x = np.arange(len(noise_order))
    offset_centre = (len(estimators) - 1) / 2.0
    for i, (est, label, colour) in enumerate(zip(estimators, labels, colours)):
        means, lows, highs = [], [], []
        for noise in noise_order:
            vals = df_delay[
                (df_delay["estimator"] == est)
                & (df_delay["noise_condition"] == noise)
            ][metric].to_numpy()
            m, lo, hi = bootstrap_mean_ci(
                vals, n_bootstrap=n_bootstrap, rng=rng,
            )
            means.append(m)
            lows.append(m - lo)
            highs.append(hi - m)
        ax.bar(
            x + (i - offset_centre) * width,
            means, width,
            yerr=[lows, highs],
            label=label, color=colour, capsize=2, alpha=0.9,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(noise_order, fontsize=9)
    ax.grid(axis="y", alpha=0.25)


def plot_per_delay_split(
    df: pd.DataFrame,
    *,
    estimators: list[str],
    labels: list[str],
    colours: list[str] | None = None,
    metric: str = "min_tip_error",
    noise_order: list[str] | None = None,
    delays: tuple[int, int] = DEFAULT_DELAYS,
    n_bootstrap: int = 10000,
    seed: int = 0,
    title: str | None = None,
    output: Path | None = None,
    figsize: tuple[float, float] = (7.6, 3.2),
    ylim: tuple[float, float] | None = (0.0, 1.0),
    success_threshold: float | None = 0.05,
) -> dict[str, float]:
    """Render the 2-subplot per-delay figure. Returns 6-cell mean summary.

    The returned dict maps estimator -> 6-cell mean of the metric so
    the caller can include it in the figure caption / paper text.
    """
    if noise_order is None:
        noise_order = DEFAULT_NOISE_ORDER
    if colours is None:
        colours = DEFAULT_COLOURS[: len(estimators)]
    if len(labels) != len(estimators) or len(colours) < len(estimators):
        raise ValueError("estimators / labels / colours must align")

    df = df.copy()
    df["delay_steps"] = df["delay_steps"].astype(int)
    rng = np.random.default_rng(seed)

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
    for ax, delay in zip(axes, delays):
        per_delay_subplot(
            ax,
            df[df["delay_steps"] == delay],
            estimators=estimators,
            labels=labels,
            colours=colours,
            noise_order=noise_order,
            metric=metric,
            n_bootstrap=n_bootstrap,
            rng=rng,
        )
        ax.set_title(f"delay $d{{=}}{delay}$", fontsize=10)
        if success_threshold is not None:
            ax.axhline(success_threshold, color="grey",
                       linestyle="--", linewidth=0.6, alpha=0.6)
        if ylim is not None:
            ax.set_ylim(*ylim)
    metric_tex = metric.replace("_", r"\_")
    axes[0].set_ylabel(f"closed-loop ${metric_tex}$ (m)")
    axes[1].legend(fontsize=8, loc="upper left", frameon=False)
    if title is not None:
        fig.suptitle(title, fontsize=10)

    # 6-cell mean per estimator -- printed for caption insertion.
    summary: dict[str, float] = {}
    for est in estimators:
        vals = df[df["estimator"] == est][metric].to_numpy()
        summary[est] = float(vals.mean()) if vals.size else float("nan")

    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, bbox_inches="tight")
        png = output.with_suffix(".png")
        fig.savefig(png, bbox_inches="tight", dpi=150)
        print(f"  wrote {output} and {png}")
    plt.close(fig)
    return summary


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metrics", type=Path, required=True)
    p.add_argument("--estimators", nargs="+", required=True)
    p.add_argument("--labels", nargs="+", required=True)
    p.add_argument("--colours", nargs="+", default=None)
    p.add_argument("--metric", default="min_tip_error")
    p.add_argument("--noise-order", nargs="+", default=None)
    p.add_argument("--delays", nargs=2, type=int, default=list(DEFAULT_DELAYS))
    p.add_argument("--n-bootstrap", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--title", default=None)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--no-success-line", action="store_true")
    args = p.parse_args()

    df = pd.read_csv(args.metrics)
    summary = plot_per_delay_split(
        df,
        estimators=args.estimators,
        labels=args.labels,
        colours=args.colours,
        metric=args.metric,
        noise_order=args.noise_order,
        delays=tuple(args.delays),
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
        title=args.title,
        output=args.output,
        success_threshold=None if args.no_success_line else 0.05,
    )
    print("6-cell mean summary (paste into caption / paper text):")
    for est, m in summary.items():
        print(f"  {est}: {m:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
