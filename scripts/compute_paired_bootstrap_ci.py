"""Compute paired percentile-bootstrap 95% CIs for estimator contrasts.

Reads a per-episode closed-loop metrics CSV (one row per
``(cell, estimator, episode_idx)``) and computes, for each requested
contrast ``A vs B`` and each cell, the paired mean difference
``A - B`` over episodes matched by ``episode_idx``.

Bootstrap design (per codex round-5b):

* ``n_bootstrap = 10000`` resamples
* stratified by cell: episodes are resampled with replacement
  independently within each cell
* matching unit: ``(cell, target_id or target_seed, episode_idx)`` -- we
  rely on ``evaluate_closed_loop`` writing the same ``episode_idx`` for
  the same target/seed combination across estimators
* CI: 2.5% and 97.5% percentiles of the bootstrap-mean-difference
  distribution
* Wilcoxon signed-rank p-value is reported as a robustness diagnostic
  but never as the primary statistic

Usage::

    uv run python scripts/compute_paired_bootstrap_ci.py \\
        --metrics runs/closed_loop/<run>/metrics.csv \\
        --contrasts default_vs_K0 global_vs_default \\
        --output runs/closed_loop/<run>/paired_stats.csv

Adds ``--include-pooled`` to additionally emit a "pooled" row per
contrast that aggregates across cells (paired across all
``(cell, episode_idx)`` rows).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    from scipy.stats import wilcoxon
except ImportError:  # SciPy is a project dependency; tolerate absence here.
    wilcoxon = None  # type: ignore[assignment]


# ----------------------------------------------------------------------
# Contrast registry. Each contrast is ``label -> (estimator_A, estimator_B)``
# where the reported ``mean_diff`` is ``A - B``. To add a new contrast
# (e.g. ``feature_cond_vs_global``) supply the matching estimator names
# present in the metrics CSV.
# ----------------------------------------------------------------------
CONTRASTS: dict[str, tuple[str, str]] = {
    "default_vs_K0": ("reliability_adaptive_v1", "K=0.0"),
    "global_vs_default": (
        "reliability_adaptive_v2_fullgrid",
        "reliability_adaptive_v1",
    ),
    # Populated when feature-conditioned / per-cell eval CSVs are merged.
    # "feature_cond_vs_global": (
    #     "feature_conditioned_beta",
    #     "reliability_adaptive_v2_fullgrid",
    # ),
    # "per_cell_vs_global": (
    #     "per_cell_beta_deployed",
    #     "reliability_adaptive_v2_fullgrid",
    # ),
}


def paired_percentile_ci(
    diffs: np.ndarray,
    *,
    n_bootstrap: int,
    rng: np.random.Generator,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Return ``(ci_low, ci_high)`` from a percentile bootstrap of the mean.

    ``diffs`` must already be paired (one number per matched pair). We
    resample with replacement at the pair level, recompute the mean,
    and take the alpha/2 and 1-alpha/2 quantiles. n=200 + percentile
    bootstrap is the codex-recommended setting; BCa is unnecessary.
    """
    n = diffs.size
    if n == 0:
        return float("nan"), float("nan")
    idx = rng.integers(0, n, size=(n_bootstrap, n))
    boot_means = diffs[idx].mean(axis=1)
    lo, hi = np.quantile(boot_means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(lo), float(hi)


def _pair_diffs(
    df: pd.DataFrame,
    est_a: str,
    est_b: str,
    *,
    metric: str,
    cell_cols: tuple[str, ...],
    pair_col: str,
) -> pd.DataFrame:
    """Build a paired-diff frame ``(cell..., pair_col, diff = A - B)``.

    Skips pairs where either side is missing.
    """
    keep_cols = list(cell_cols) + [pair_col, "estimator", metric]
    sub = df[df["estimator"].isin([est_a, est_b])][keep_cols].copy()
    wide = sub.pivot_table(
        index=list(cell_cols) + [pair_col],
        columns="estimator",
        values=metric,
        aggfunc="first",
    ).reset_index()
    wide = wide.dropna(subset=[est_a, est_b])
    wide["diff"] = wide[est_a] - wide[est_b]
    return wide


def compute_contrast(
    df: pd.DataFrame,
    label: str,
    est_a: str,
    est_b: str,
    *,
    metric: str,
    cell_cols: tuple[str, ...],
    pair_col: str,
    n_bootstrap: int,
    seed: int,
    include_pooled: bool,
) -> list[dict]:
    """Compute per-cell (and optional pooled) paired stats for one contrast.

    Returns a list of records suitable for a long-form CSV.
    """
    rng = np.random.default_rng(seed)
    pairs = _pair_diffs(
        df, est_a, est_b, metric=metric, cell_cols=cell_cols, pair_col=pair_col,
    )
    records: list[dict] = []

    for cell_key, group in pairs.groupby(list(cell_cols)):
        diffs = group["diff"].to_numpy()
        lo, hi = paired_percentile_ci(
            diffs, n_bootstrap=n_bootstrap, rng=rng,
        )
        rec = {
            "contrast": label,
            "estimator_a": est_a,
            "estimator_b": est_b,
            "metric": metric,
            "n_pairs": int(diffs.size),
            "mean_diff": float(diffs.mean()) if diffs.size else float("nan"),
            "ci_low": lo,
            "ci_high": hi,
        }
        for col, val in zip(cell_cols, cell_key if isinstance(cell_key, tuple) else (cell_key,)):
            rec[col] = val
        if wilcoxon is not None and diffs.size >= 2 and np.any(diffs != 0):
            try:
                rec["wilcoxon_p"] = float(wilcoxon(diffs).pvalue)
            except ValueError:
                rec["wilcoxon_p"] = float("nan")
        else:
            rec["wilcoxon_p"] = float("nan")
        records.append(rec)

    if include_pooled and not pairs.empty:
        diffs = pairs["diff"].to_numpy()
        lo, hi = paired_percentile_ci(
            diffs, n_bootstrap=n_bootstrap, rng=rng,
        )
        rec = {
            "contrast": label,
            "estimator_a": est_a,
            "estimator_b": est_b,
            "metric": metric,
            "n_pairs": int(diffs.size),
            "mean_diff": float(diffs.mean()),
            "ci_low": lo,
            "ci_high": hi,
            "wilcoxon_p": (
                float(wilcoxon(diffs).pvalue)
                if wilcoxon is not None and diffs.size >= 2 and np.any(diffs != 0)
                else float("nan")
            ),
        }
        for col in cell_cols:
            rec[col] = "POOLED"
        records.append(rec)
    return records


def main(argv: Iterable[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metrics", type=Path, required=True,
                   help="Per-episode metrics CSV from evaluate_closed_loop.py")
    p.add_argument("--output", type=Path, required=True,
                   help="Output CSV: long-form paired_stats")
    p.add_argument("--contrasts", nargs="+", default=list(CONTRASTS),
                   help="Contrast labels to compute (subset of CONTRASTS keys)")
    p.add_argument("--metric", default="min_tip_error")
    p.add_argument("--cell-cols", nargs="+",
                   default=["noise_condition", "delay_steps"])
    p.add_argument("--pair-col", default="episode_idx",
                   help="Per-episode key whose value is shared across estimators")
    p.add_argument("--n-bootstrap", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--include-pooled", action="store_true",
                   help="Add a pooled-across-cells row per contrast")
    args = p.parse_args(list(argv) if argv is not None else None)

    df = pd.read_csv(args.metrics)
    needed = set(args.cell_cols) | {args.pair_col, "estimator", args.metric}
    missing = needed - set(df.columns)
    if missing:
        raise SystemExit(f"metrics CSV is missing columns: {sorted(missing)}")

    all_records: list[dict] = []
    for label in args.contrasts:
        if label not in CONTRASTS:
            raise SystemExit(
                f"unknown contrast {label!r}; known: {sorted(CONTRASTS)}"
            )
        est_a, est_b = CONTRASTS[label]
        if not {est_a, est_b}.issubset(df["estimator"].unique()):
            print(f"  skipping {label}: estimators not in CSV")
            continue
        all_records.extend(
            compute_contrast(
                df, label, est_a, est_b,
                metric=args.metric,
                cell_cols=tuple(args.cell_cols),
                pair_col=args.pair_col,
                n_bootstrap=args.n_bootstrap,
                seed=args.seed,
                include_pooled=args.include_pooled,
            )
        )

    out = pd.DataFrame.from_records(all_records)
    # Stable column order.
    leading = [
        "contrast", "estimator_a", "estimator_b", "metric",
        *args.cell_cols, "n_pairs", "mean_diff", "ci_low", "ci_high",
        "wilcoxon_p",
    ]
    out = out[[c for c in leading if c in out.columns]]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"  wrote {args.output} ({len(out)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
