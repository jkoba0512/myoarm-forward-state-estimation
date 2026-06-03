"""Precompute per-cell β JSONs from a feature-conditioned W and base β.

Stage C C4 deploys the feature-conditioned β per cell by computing:

    β_field = base_β_field + W[field, side, 0] * cell_mean_field
                           + W[field, side, 1] * cell_var_field
                           + W[field, side, 2]

This script materialises the 6 per-cell β JSONs ahead of time so the
deployment can go through ``scripts/evaluate_closed_loop.py`` with a
plain ``beta_source`` pointer per cell, keeping the metrics.csv schema
unified with C1-C3 / per-cell deploys (Codex 2026-06-01 Q27 + schema
unification requirement).

Output dir is structured as:

    <out_dir>/feature_conditioned_beta_cells/
      cell_none_d0.json
      cell_none_d18.json
      cell_high_d0.json
      cell_high_d18.json
      cell_xhigh_d0.json
      cell_xhigh_d18.json
      meta.json                   # provenance: W / base_β / features paths

Usage:
    uv run python scripts/precompute_feature_conditioned_betas.py \\
        --w-source runs/feature_conditioned_beta/<eval>/final_W.json \\
        --base-beta runs/reliability_adaptive_v2/<eval>/final_beta.json \\
        --cell-features runs/diagnostics/feature_conditioned/cell_features_r3_v2.json \\
        --output runs/feature_conditioned_beta/<eval>/per_cell
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

FIELDS = ("qpos", "qvel", "act", "tip_pos", "reach_err")
CELLS = [
    ("none", 0), ("none", 18),
    ("high", 0), ("high", 18),
    ("xhigh", 0), ("xhigh", 18),
]


def compute_beta(W, base_beta0, base_beta1, cell_features):
    """Apply per-cell adjustment to base β. Mirrors the formula from
    ``eval_feature_conditioned_beta.py`` (5 fields × 2 sides × 3 features)."""
    beta0 = dict(base_beta0)
    beta1 = dict(base_beta1)
    for i, f in enumerate(FIELDS):
        x_mean = cell_features[f"mean_{f}"]
        x_var = cell_features[f"var_{f}"]
        db0 = W[i, 0, 0] * x_mean + W[i, 0, 1] * x_var + W[i, 0, 2]
        db1 = W[i, 1, 0] * x_mean + W[i, 1, 1] * x_var + W[i, 1, 2]
        beta0[f] = base_beta0[f] + db0
        beta1[f] = base_beta1[f] + db1
    return beta0, beta1


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--w-source", type=Path, required=True,
                   help="path to feature_conditioned final_W.json (B.4)")
    p.add_argument("--base-beta", type=Path, required=True,
                   help="path to fullgrid final_beta.json (B.3)")
    p.add_argument("--cell-features", type=Path, required=True,
                   help="path to cell_features_r3_v2.json (B.1)")
    p.add_argument("--output", type=Path, required=True,
                   help="output directory for per-cell β JSONs")
    args = p.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    W_blob = json.load(open(args.w_source))
    W = np.array(W_blob["W_flat"]).reshape(5, 2, 3)

    base = json.load(open(args.base_beta))
    base_beta0 = {f: float(base["beta0"][f]) for f in FIELDS}
    base_beta1 = {f: float(base["beta1"][f]) for f in FIELDS}

    cell_features_blob = json.load(open(args.cell_features))
    # compute_per_cell_features.py wraps the per-cell dicts under "raw"
    # alongside z-scored/normalised variants; we want the raw values.
    raw_cells = cell_features_blob.get("raw", cell_features_blob)

    written: list[str] = []
    for noise, delay in CELLS:
        cell_key = f"{noise}_d{delay}"
        if cell_key not in raw_cells:
            raise SystemExit(
                f"cell_features.raw missing key {cell_key!r}; "
                f"available: {list(raw_cells.keys())[:6]}"
            )
        cf = raw_cells[cell_key]

        beta0, beta1 = compute_beta(W, base_beta0, base_beta1, cf)

        out_path = args.output / f"cell_{cell_key}.json"
        out_path.write_text(json.dumps({
            "beta0": {f: float(beta0[f]) for f in FIELDS},
            "beta1": {f: float(beta1[f]) for f in FIELDS},
            "alpha": float(base.get("alpha", 0.05)),
            "epsilon": float(base.get("epsilon", 1e-6)),
            "var_init": float(base.get("var_init", 1.0)),
            "target_pos_gain": float(base.get("target_pos_gain", 1.0)),
        }, indent=2))
        written.append(str(out_path))
        print(f"  {cell_key}: β written → {out_path}")

    meta = {
        "w_source": str(args.w_source),
        "base_beta_source": str(args.base_beta),
        "cell_features_source": str(args.cell_features),
        "cells": [f"{n}_d{d}" for n, d in CELLS],
        "written_betas": written,
    }
    (args.output / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nmeta saved: {args.output / 'meta.json'}")


if __name__ == "__main__":
    main()
