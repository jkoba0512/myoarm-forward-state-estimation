"""Stage C paired bootstrap CI across the R3 deploy artifacts.

Reads per-episode metrics from C1 / C2 / C3 / C4 / per-cell deploy
runs (plus B.6 K-sweep) and computes, per cell, the paired mean
difference ``A - B`` over matched ``episode_id`` together with a
percentile-bootstrap 95 % CI. Codex 2026-06-01 (Stage B gate response):

  key contrasts:
    default reliability  vs  best fixed K
    global β             vs  default reliability
    feature-conditioned  vs  global β
    feature-conditioned  vs  per-cell β diagnostic
    feature-conditioned  vs  oracle best K
    K=0                  vs  best K        (diagnostic)

This script is intentionally minimal and self-contained so we do not
need to retrofit ``compute_paired_bootstrap_ci.py`` to the new R3
estimator names and per-cell-split CSV layout.

Usage:
    uv run python scripts/stage_c_paired_ci.py \\
        --c1 runs/closed_loop/<c1>/metrics.csv \\
        --c2 runs/closed_loop/<c2>/metrics.csv \\
        --c3 runs/closed_loop/<c3>/metrics.csv \\
        --c4-dir runs/closed_loop/  \\
        --c4-prefix c4_feature_conditioned \\
        --per-cell-dir runs/closed_loop/ \\
        --per-cell-prefix per_cell_beta \\
        --b6 runs/closed_loop/2026-05-27T11-09-25Z/metrics.csv \\
        --output runs/diag/r3_stage_c_paired_ci.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np

CELLS = [
    ("none", 0), ("none", 18),
    ("high", 0), ("high", 18),
    ("xhigh", 0), ("xhigh", 18),
]


def _load_metrics(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def _filter_min_tip(rows: list[dict], cell_noise: str, cell_delay: int,
                    estimator: str | None = None) -> dict[int, float]:
    """Return {episode_id: min_tip} for one (cell, estimator)."""
    out: dict[int, float] = {}
    for r in rows:
        if r["noise_condition"] != cell_noise:
            continue
        if int(r["delay_steps"]) != cell_delay:
            continue
        if estimator is not None and r["estimator"] != estimator:
            continue
        out[int(r["episode_id"])] = float(r["min_tip_error"])
    return out


def _paired_bootstrap(diffs: np.ndarray, n_boot: int = 10_000,
                      seed: int = 0) -> tuple[float, float, float]:
    """Return (mean_diff, ci_lo, ci_hi) for a percentile bootstrap."""
    rng = np.random.default_rng(seed)
    n = diffs.shape[0]
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    idx = rng.integers(0, n, size=(n_boot, n))
    means = diffs[idx].mean(axis=1)
    ci_lo = float(np.percentile(means, 2.5))
    ci_hi = float(np.percentile(means, 97.5))
    return float(diffs.mean()), ci_lo, ci_hi


def _b6_best_K_min_tip(b6_rows: list[dict], cell_noise: str,
                      cell_delay: int) -> tuple[str, dict[int, float]]:
    """Return ('K=X.XX', {episode_id: min_tip}) for the per-cell best K
    (argmin of cell-mean min_tip across the K-grid)."""
    by_est: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for r in b6_rows:
        if (r["noise_condition"] == cell_noise
                and int(r["delay_steps"]) == cell_delay):
            by_est[r["estimator"]].append(
                (int(r["episode_id"]), float(r["min_tip_error"]))
            )
    if not by_est:
        return "", {}
    cell_means = {est: float(np.mean([v for _, v in eps]))
                  for est, eps in by_est.items()}
    best = min(cell_means, key=cell_means.get)
    return best, dict(by_est[best])


def _b6_K0_min_tip(b6_rows: list[dict], cell_noise: str,
                   cell_delay: int) -> dict[int, float]:
    return _filter_min_tip(b6_rows, cell_noise, cell_delay, "K=0.00")


def _load_per_cell_dir(out_root: Path, prefix: str,
                      cell_noise: str, cell_delay: int) -> dict[int, float]:
    """For C4 / per-cell batch outputs, each cell lives under a separate
    closed_loop/<eval_id>/metrics.csv. Walk recent dirs and pick the one
    whose estimator matches the cell prefix."""
    label = f"{cell_noise}_d{cell_delay}"
    needle = f"{prefix}_{label}"
    candidates = sorted(out_root.glob("2026-*/metrics.csv"), reverse=True)
    for csv_path in candidates:
        # estimator column lookup
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            first = next(reader, None)
        if first is None:
            continue
        if first.get("estimator") == needle:
            return _filter_min_tip(
                _load_metrics(csv_path), cell_noise, cell_delay, needle,
            )
    return {}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--c1", type=Path, required=True)
    p.add_argument("--c2", type=Path, required=True)
    p.add_argument("--c3", type=Path, required=True)
    p.add_argument("--c4-dir", type=Path, required=True)
    p.add_argument("--c4-prefix", default="c4_feature_conditioned")
    p.add_argument("--per-cell-dir", type=Path, required=True)
    p.add_argument("--per-cell-prefix", default="per_cell_beta")
    p.add_argument("--b6", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--n-boot", type=int, default=10_000)
    args = p.parse_args()

    c1 = _load_metrics(args.c1)
    c2 = _load_metrics(args.c2)
    c3 = _load_metrics(args.c3)
    b6 = _load_metrics(args.b6)
    print(f"loaded: c1={len(c1)} c2={len(c2)} c3={len(c3)} b6={len(b6)} rows")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_rows: list[dict] = []

    for noise, delay in CELLS:
        # Per-cell datasets (episode_id-keyed dicts)
        d_c1 = _filter_min_tip(c1, noise, delay, "default_reliability")
        d_c2 = (_filter_min_tip(c2, noise, delay, "c2_single_cell_beta")
                if (noise == "none" and delay == 18) else {})
        d_c3 = _filter_min_tip(c3, noise, delay, "c3_global_beta")
        d_c4 = _load_per_cell_dir(args.c4_dir, args.c4_prefix, noise, delay)
        d_pc = _load_per_cell_dir(
            args.per_cell_dir, args.per_cell_prefix, noise, delay,
        )
        best_K_label, d_bestK = _b6_best_K_min_tip(b6, noise, delay)
        d_K0 = _b6_K0_min_tip(b6, noise, delay)

        def _diff(label, a_d, b_d, a_lbl, b_lbl):
            if not a_d or not b_d:
                return None
            common = sorted(set(a_d) & set(b_d))
            if not common:
                return None
            diffs = np.array([a_d[k] - b_d[k] for k in common])
            mean, lo, hi = _paired_bootstrap(
                diffs, n_boot=args.n_boot, seed=0,
            )
            return {
                "contrast": label,
                "cell_noise": noise, "cell_delay": delay,
                "estimator_a": a_lbl, "estimator_b": b_lbl,
                "n": len(common),
                "mean_diff": mean, "ci_lo": lo, "ci_hi": hi,
                "metric": "min_tip_error",
            }

        contrasts = [
            ("default_vs_bestK", d_c1, d_bestK,
             "default_reliability", f"B.6 bestK ({best_K_label})"),
            ("global_vs_default", d_c3, d_c1,
             "c3_global_beta", "default_reliability"),
            ("feature_cond_vs_global", d_c4, d_c3,
             "c4_feature_conditioned", "c3_global_beta"),
            ("feature_cond_vs_per_cell", d_c4, d_pc,
             "c4_feature_conditioned", "per_cell_beta"),
            ("feature_cond_vs_bestK", d_c4, d_bestK,
             "c4_feature_conditioned", f"B.6 bestK ({best_K_label})"),
            ("K0_vs_bestK", d_K0, d_bestK,
             "K=0.00", f"B.6 bestK ({best_K_label})"),
            ("per_cell_vs_bestK", d_pc, d_bestK,
             "per_cell_beta", f"B.6 bestK ({best_K_label})"),
            # C2 only meaningful at (none, 18)
            ("c2_single_vs_default", d_c2, d_c1,
             "c2_single_cell_beta", "default_reliability"),
        ]

        for label, a, b, a_lbl, b_lbl in contrasts:
            row = _diff(label, a, b, a_lbl, b_lbl)
            if row is not None:
                out_rows.append(row)

    # write
    cols = ["contrast", "cell_noise", "cell_delay",
            "estimator_a", "estimator_b", "n",
            "mean_diff", "ci_lo", "ci_hi", "metric"]
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in out_rows:
            w.writerow(r)
    print(f"saved {len(out_rows)} rows → {args.output}")

    # Also print a short summary by contrast.
    by_contrast: dict[str, list[dict]] = defaultdict(list)
    for r in out_rows:
        by_contrast[r["contrast"]].append(r)
    print("\n=== summary by contrast (per cell mean_diff ± [CI]) ===")
    for label, rows in by_contrast.items():
        print(f"\n[{label}]")
        for r in rows:
            md, lo, hi = r["mean_diff"], r["ci_lo"], r["ci_hi"]
            sig = "*" if (lo > 0 or hi < 0) else " "
            print(f"  ({r['cell_noise']:>5}, d={r['cell_delay']:>2}) "
                  f"n={r['n']:>3}  Δ={md:+.4f}  "
                  f"[{lo:+.4f}, {hi:+.4f}] {sig}")


if __name__ == "__main__":
    main()
