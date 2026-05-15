"""Plot the RQ1 geometry regression result as F_geometry_regression.

Two-panel figure:
  Left:  Delta-outcome (K=1 - K=0) vs reliability_variance
         (alone, R^2 = 0.22 — reliability is insufficient)
  Right: Delta-outcome vs fm_rollout_mse_h10
         (alone, R^2 strong — forward-model error is the dominant factor)

Each panel shows the 18 (model x cell) data points colored by model H,
the simple linear fit, and the per-panel R^2 annotation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FIG_DIR = Path("figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

MODEL_LABEL = {
    "2026-05-10T07-25-23Z": "$H{=}1$",
    "2026-05-11T09-53-07Z": "$H{=}4$",
    "2026-05-11T11-47-34Z": "$H{=}8$",
}
MODEL_COLOR = {
    "2026-05-10T07-25-23Z": "#d62728",  # red — worst
    "2026-05-11T09-53-07Z": "#ff7f0e",  # orange — middle
    "2026-05-11T11-47-34Z": "#1f77b4",  # blue — best
}


def _r2(x, y):
    if x.std() < 1e-12 or y.std() < 1e-12:
        return float("nan")
    c = np.corrcoef(x, y)[0, 1]
    return float(c ** 2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path,
                   default=Path("runs/diagnostics/geometry/rq1_regression.csv"))
    p.add_argument("--out-stem", type=str, default="F_geometry_regression")
    args = p.parse_args()

    df = pd.read_csv(args.data)
    df = df.dropna(subset=["reliability_variance", "fm_rollout_mse_h10",
                            "delta_outcome"])

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.6))
    for ax, predictor, xlabel in [
        (axes[0], "reliability_variance",
         "Reliability variance ($\\sum_f v_f$)"),
        (axes[1], "fm_rollout_mse_h10",
         "Forward-model rollout error MSE ($h{=}10$)"),
    ]:
        for model_id, sub in df.groupby("model_id"):
            ax.scatter(sub[predictor], sub["delta_outcome"],
                       c=MODEL_COLOR[model_id], s=70,
                       label=MODEL_LABEL[model_id],
                       edgecolors="black", linewidths=0.4, alpha=0.85)
        x = df[predictor].values
        y = df["delta_outcome"].values
        r2 = _r2(x, y)
        if not np.isnan(r2):
            xf = np.linspace(x.min(), x.max(), 100)
            slope, intercept = np.polyfit(x, y, 1)
            ax.plot(xf, slope * xf + intercept,
                    color="grey", linestyle="--", linewidth=1.0)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("$\\Delta$outcome $= E[\\min$-tip$\\,|\\,K{=}1]"
                      " - E[\\min$-tip$\\,|\\,K{=}0]$ (m)")
        ax.axhline(0, color="black", linewidth=0.4, alpha=0.5)
        ax.grid(alpha=0.25)
        ax.set_title(f"$R^2 = {r2:.3f}$" if not np.isnan(r2) else "")
        ax.legend(fontsize=8, frameon=False, loc="upper left"
                  if predictor == "reliability_variance" else "upper right")

    fig.suptitle("RQ1: $\\Delta$outcome is dominated by forward-model "
                 "error, not by sensory reliability", fontsize=11)
    fig.tight_layout()
    pdf = FIG_DIR / f"{args.out_stem}.pdf"
    png = FIG_DIR / f"{args.out_stem}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight", dpi=150)
    print(f"wrote {pdf} and {png}")


if __name__ == "__main__":
    main()
