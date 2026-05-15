"""RQ1: regress Delta-outcome(K=1 - K=0) on reliability + delay + forward-model
error + forward-model bias, per (cell, forward-model).

Primary target: Delta outcome (continuous, m)
Predictors:
  - delay_steps                                  (0 or 18)
  - reliability_variance       sum over fields of EMA innovation variance
  - fm_rollout_mse_h10         midhorizon rollout error
  - fm_tip_err_h10             midhorizon tip error
  - fm_bias_norm               aggregate bias norm
  - fm_tip_signed_bias         task-space signed bias

Output:
  runs/diagnostics/geometry/rq1_regression.csv  (per-cell, per-model data)
  runs/diagnostics/geometry/rq1_coefficients.csv  (regression coefficients + R^2)
  Two simple ASCII tables to stdout

Usage::
    uv run python scripts/rq1_geometry_regression.py
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _fit_ols(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    """Plain OLS via lstsq + R^2. Returns (coefs, r2)."""
    X_std = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-12)
    X1 = np.concatenate([X_std, np.ones((X_std.shape[0], 1))], axis=1)
    coefs, residuals, rank, _ = np.linalg.lstsq(X1, y, rcond=None)
    y_pred = X1 @ coefs
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")
    return coefs[:-1], r2  # drop intercept


CELLS = [("none", 0), ("none", 18), ("high", 0), ("high", 18),
         ("xhigh", 0), ("xhigh", 18)]


def load_fm_diag(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def load_closed_loop_kvals(closed_loop_dir: Path) -> pd.DataFrame:
    """Aggregate K=0 / K=1 / K=0.25 / K=0.5 / K=0.75 min-tip per (cell, model).

    Walk closed_loop_dir, find runs whose config.estimators include the
    fixed-gain estimators, extract per-cell means.
    """
    rows = []
    for run_dir in closed_loop_dir.iterdir():
        if not run_dir.is_dir():
            continue
        cfg_path = run_dir / "config.json"
        metrics_path = run_dir / "metrics.csv"
        if not cfg_path.exists() or not metrics_path.exists():
            continue
        cfg = json.load(open(cfg_path))
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


def aggregate_kvals(df: pd.DataFrame) -> pd.DataFrame:
    """Per (model, cell), median min-tip across all duplicates for each K."""
    agg = (
        df.groupby(["model_id", "noise_condition", "delay_steps", "estimator"])[
            "min_tip_error"
        ]
        .agg("mean")
        .reset_index()
    )
    pivot = agg.pivot_table(
        index=["model_id", "noise_condition", "delay_steps"],
        columns="estimator",
        values="min_tip_error",
    ).reset_index()
    pivot.columns.name = None
    return pivot


def add_reliability_variance(rows: pd.DataFrame, diag_dir: Path) -> pd.DataFrame:
    """Pull reliability variance per cell from the default-reliability dump.

    Only valid for H=8 reachfixed (`2026-05-11T11-47-34Z` model). For
    other models we set NaN (will be imputed or dropped).
    """
    rel_var = {}
    for noise in ("none", "high", "xhigh"):
        for delay in (0, 18):
            f = diag_dir / f"default_{noise}_d{delay}.npz"
            if f.exists():
                d = np.load(f)
                T = d["k_qpos"].shape[0]
                half = T // 2
                # sum of EMA innovation variance across 5 fields
                total_var = 0.0
                for field in ("qpos", "qvel", "act", "tip_pos", "reach_err"):
                    v = d[f"var_{field}"][half:]
                    total_var += float(v.mean())
                rel_var[(noise, delay)] = total_var

    def lookup(row):
        return rel_var.get((row["noise"], row["delay"]), np.nan)

    rows["reliability_variance"] = rows.apply(lookup, axis=1)
    return rows


def main() -> Path:
    p = argparse.ArgumentParser()
    p.add_argument("--fm-diag", type=Path,
                   default=Path("runs/diagnostics/geometry/fm_diag.csv"))
    p.add_argument("--closed-loop-dir", type=Path,
                   default=Path("runs/closed_loop"))
    p.add_argument("--reliability-diag-dir", type=Path,
                   default=Path("runs/diagnostics"))
    p.add_argument("--output-dir", type=Path,
                   default=Path("runs/diagnostics/geometry"))
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Forward-model diagnostics (per cell × model)
    fm = load_fm_diag(args.fm_diag)
    print(f"FM diag: {len(fm)} rows")

    # 2. Closed-loop K-value outcomes (per cell × model × estimator)
    cl = load_closed_loop_kvals(args.closed_loop_dir)
    kvals = aggregate_kvals(cl)
    print(f"K-vals: {len(kvals)} rows")
    print("models with K-sweep:", sorted(kvals["model_id"].unique()))

    # 3. Combine
    fm_rename = fm.rename(columns={"model_id": "model_id",
                                   "noise": "noise_condition",
                                   "delay": "delay_steps"})
    combined = kvals.merge(fm_rename,
                           on=["model_id", "noise_condition", "delay_steps"],
                           how="inner")
    print(f"Combined: {len(combined)} rows")

    # 4. Add reliability variance (rename for compatibility)
    combined = combined.rename(columns={"noise_condition": "noise",
                                        "delay_steps": "delay"})
    combined = add_reliability_variance(combined, args.reliability_diag_dir)

    # 5. Compute Delta outcome = K=1 - K=0
    if "K=0.0" not in combined.columns or "K=1.0" not in combined.columns:
        print("ERROR: K=0.0 or K=1.0 column missing from combined data")
        print("Available columns:", combined.columns.tolist())
        return args.output_dir

    combined["delta_outcome"] = combined["K=1.0"] - combined["K=0.0"]
    out_csv = args.output_dir / "rq1_regression.csv"
    combined.to_csv(out_csv, index=False)
    print(f"Saved {out_csv}")
    print(combined[["model_id", "noise", "delay", "K=0.0", "K=1.0",
                    "delta_outcome", "reliability_variance",
                    "fm_rollout_mse_h10", "fm_bias_norm"]].to_string(
        index=False, float_format="%.3f"))

    # 6. Regression: 2 models
    #   (a) reliability + delay only        ← baseline
    #   (b) reliability + delay + fm-error + fm-bias  ← full

    # Drop rows with NaN reliability (non-H=8 model)
    reg_data = combined.dropna(subset=["reliability_variance",
                                        "fm_rollout_mse_h10",
                                        "fm_bias_norm",
                                        "fm_tip_signed_bias"])
    print(f"\nRegression data: {len(reg_data)} rows")
    if len(reg_data) < 4:
        print("Insufficient rows for regression.")
        return out_csv

    y = reg_data["delta_outcome"].values
    Xs = {
        "reliability_only": reg_data[["reliability_variance", "delay"]].values,
        "fm_only":          reg_data[["fm_rollout_mse_h10",
                                       "fm_bias_norm",
                                       "fm_tip_signed_bias",
                                       "delay"]].values,
        "full":             reg_data[["reliability_variance",
                                       "fm_rollout_mse_h10",
                                       "fm_bias_norm",
                                       "fm_tip_signed_bias",
                                       "delay"]].values,
    }
    feat_names = {
        "reliability_only": ["reliability_var", "delay"],
        "fm_only":          ["fm_h10_mse", "fm_bias_norm",
                              "fm_tip_signed_bias", "delay"],
        "full":             ["reliability_var", "fm_h10_mse", "fm_bias_norm",
                              "fm_tip_signed_bias", "delay"],
    }

    print("\n=== Regression R^2 ===")
    coef_rows = []
    for name, X in Xs.items():
        coef, r2 = _fit_ols(X, y)
        print(f"  {name:18s}: R^2 = {r2:.3f}, coefs = "
              f"{[f'{c:+.3f}' for c in coef]}")
        for fname, c in zip(feat_names[name], coef):
            coef_rows.append({
                "model": name, "feature": fname, "std_coef": c,
            })
        coef_rows.append({
            "model": name, "feature": "_R2_", "std_coef": r2,
        })

    coef_csv = args.output_dir / "rq1_coefficients.csv"
    pd.DataFrame(coef_rows).to_csv(coef_csv, index=False)
    print(f"\nSaved {coef_csv}")
    return out_csv


if __name__ == "__main__":
    main()
