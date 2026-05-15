"""Plot the RQ2 forward-model quality axis result as F_fm_quality_shift.

Aggregates K-sweep results across {H=1, H=4, H=8, undertrained H=8}
and shows how K*_cl (the per-cell closed-loop optimum K) shifts with
forward-model quality.

Two-panel figure:
  Left:  per-cell min-tip vs K for each model (line plot, 4 lines per
         cell × 6 cells, faceted by cell)
  Right: K*_cl distribution across cells, per model (boxplot or bars)

If undertrained H=8 isn't done yet, falls back to 3 models.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FIG_DIR = Path("figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

CELLS = [("none", 0), ("none", 18), ("high", 0), ("high", 18),
         ("xhigh", 0), ("xhigh", 18)]

# Model ID -> human label, ordered from "worst" to "best".
MODEL_LABEL_ORDER = [
    ("2026-05-10T07-25-23Z", "$H{=}1$"),
    ("2026-05-11T09-53-07Z", "$H{=}4$"),
    # undertrained H=8 is inserted at runtime if present
    ("2026-05-11T11-47-34Z", "$H{=}8$"),
]
MODEL_COLOR = {
    "$H{=}1$": "#d62728",
    "$H{=}4$": "#ff7f0e",
    "$H{=}8$ undertrained": "#9467bd",
    "$H{=}8$": "#1f77b4",
}


def collect_kvals(closed_loop_dir: Path) -> pd.DataFrame:
    """Walk runs and aggregate fixed-gain K-sweep per (model, cell, K)."""
    rows = []
    for run_dir in closed_loop_dir.iterdir():
        if not run_dir.is_dir():
            continue
        cfg_path = run_dir / "config.json"
        metrics_path = run_dir / "metrics.csv"
        if not cfg_path.exists() or not metrics_path.exists():
            continue
        try:
            cfg = json.load(open(cfg_path))
        except Exception:
            continue
        c = cfg.get("config", cfg)
        if c.get("env_id") != "myoArmReachFixed-v0":
            continue
        fm_path = c.get("forward_model", "")
        model_id = Path(fm_path).name if fm_path else "?"
        ests = c.get("estimators", [])
        est_names = [e.get("name") for e in ests if e.get("kind") == "fixed"]
        if not est_names:
            continue
        try:
            df = pd.read_csv(metrics_path)
        except Exception:
            continue
        df = df[df["estimator"].isin(est_names)]
        df["model_id"] = model_id
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--closed-loop-dir", type=Path,
                   default=Path("runs/closed_loop"))
    p.add_argument("--undertrained-h8-model", type=str, default=None,
                   help="If provided, include this model id as H=8 undertrained")
    p.add_argument("--out-stem", type=str, default="F_fm_quality_shift")
    args = p.parse_args()

    df = collect_kvals(args.closed_loop_dir)
    if df.empty:
        print("No K-sweep data found.")
        return

    # Extract K from estimator name "K=0.25" etc.
    df["K"] = df["estimator"].str.extract(r"K=([\d\.]+)").astype(float)
    df = df.dropna(subset=["K"])

    model_label_order = list(MODEL_LABEL_ORDER)
    if args.undertrained_h8_model is not None:
        model_label_order.insert(
            2, (args.undertrained_h8_model, "$H{=}8$ undertrained"))

    df = df[df["model_id"].isin([m for m, _ in model_label_order])].copy()
    label_map = dict(model_label_order)
    df["model_label"] = df["model_id"].map(label_map)

    agg = (
        df.groupby(["model_label", "noise_condition", "delay_steps", "K"])[
            "min_tip_error"
        ].mean().reset_index()
    )

    fig, axes = plt.subplots(2, 3, figsize=(11, 6), sharey=True, sharex=True)
    for (noise, delay), ax in zip(CELLS, axes.flatten()):
        for _, label in model_label_order:
            row = agg[(agg["model_label"] == label) &
                       (agg["noise_condition"] == noise) &
                       (agg["delay_steps"] == delay)].sort_values("K")
            if row.empty:
                continue
            ax.plot(row["K"], row["min_tip_error"], "o-",
                    color=MODEL_COLOR[label], label=label,
                    linewidth=1.6, markersize=5)
        ax.set_title(f"({noise}, $d{{=}}{delay}$)", fontsize=10)
        ax.grid(alpha=0.25)
        ax.axhline(0.05, color="grey", linestyle=":", linewidth=0.6, alpha=0.5)

    axes[0, 0].set_ylabel("min-tip error (m)")
    axes[1, 0].set_ylabel("min-tip error (m)")
    for ax in axes[1, :]:
        ax.set_xlabel("correction gain $K$")
    axes[0, 0].legend(fontsize=8, frameon=False, loc="upper right")
    fig.suptitle("RQ2: closed-loop K-sweep across forward-model quality",
                 fontsize=11)
    fig.tight_layout()
    pdf = FIG_DIR / f"{args.out_stem}.pdf"
    png = FIG_DIR / f"{args.out_stem}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight", dpi=150)
    print(f"wrote {pdf} and {png}")


if __name__ == "__main__":
    main()
