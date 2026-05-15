"""Run 6 independent SPSA training runs, one per focused-grid cell.

Each cell c = (noise, delay) gets its own β trained for 100 iterations
on that cell only. The 6 β configurations are the per-cell diagnostic
ceiling for RQ3 — they are deliberately NOT positioned as a biological
mechanism.

This driver generates 6 temporary configs (one per cell), invokes the
main SPSA training script on each, and collects summaries.

Outputs (per cell, under <output-root>/<eval_id>/):
  cell_<label>/<spsa-eval-id>/final_beta.json
  cell_<label>/<spsa-eval-id>/history.json
  summary.csv

Walltime: ~100 min (6 cells × 100 iter × 20 episodes × ~0.5 s/ep).
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml


NOISE_PRESETS = {
    "none":  {"qpos": 0.0,  "qvel": 0.0,  "tip_pos": 0.0,  "reach_err": 0.0},
    "high":  {"qpos": 0.02, "qvel": 0.02, "tip_pos": 0.01, "reach_err": 0.01},
    "xhigh": {"qpos": 0.08, "qvel": 0.08, "tip_pos": 0.04, "reach_err": 0.04},
}

CELLS = [(n, d) for d in (0, 18) for n in ("none", "high", "xhigh")]


def build_per_cell_config(noise: str, delay: int, base_cfg: dict,
                          max_iter: int, samples: int,
                          a: float, c: float, A: float,
                          output_root: Path) -> dict:
    """Return a fully-resolved single-cell SPSA config."""
    cfg = json.loads(json.dumps(base_cfg))  # deep copy
    cfg["noise_conditions"] = {noise: NOISE_PRESETS[noise]}
    cfg["delay_grid"] = [delay]
    cfg["spsa"]["max_iter"] = max_iter
    cfg["spsa"]["samples_per_side"] = samples
    cfg["spsa"]["a"] = a
    cfg["spsa"]["c"] = c
    cfg["spsa"]["A"] = A
    cfg["output_root"] = str(output_root)
    return cfg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-config", type=Path,
                   default=Path("configs/train/reliability_adaptive_v2_poc.yaml"))
    p.add_argument("--output-root", type=Path,
                   default=Path("runs/per_cell_beta_diagnostic"))
    p.add_argument("--max-iter", type=int, default=100)
    p.add_argument("--samples-per-side", type=int, default=10)
    p.add_argument("--a", type=float, default=2.0)
    p.add_argument("--c", type=float, default=0.3)
    p.add_argument("--A", type=float, default=5.0)
    args = p.parse_args()

    eval_id = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    out_root = args.output_root / eval_id
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {out_root}")

    base_cfg = yaml.safe_load(open(args.base_config))

    rows = []
    for noise, delay in CELLS:
        cell_label = f"{noise}_d{delay}"
        cell_dir = out_root / f"cell_{cell_label}"
        cell_dir.mkdir(parents=True, exist_ok=True)
        cfg = build_per_cell_config(
            noise, delay, base_cfg,
            args.max_iter, args.samples_per_side,
            args.a, args.c, args.A,
            cell_dir,
        )
        cfg_path = cell_dir / "config.yaml"
        with open(cfg_path, "w") as f:
            yaml.safe_dump(cfg, f)

        print(f"\n==== cell {cell_label} ====  config: {cfg_path}")
        t0 = time.time()
        result = subprocess.run(
            ["uv", "run", "python", "scripts/train_reliability_adaptive_v2.py",
             "--config", str(cfg_path)],
            capture_output=True, text=True,
        )
        elapsed = time.time() - t0
        if result.returncode != 0:
            print(f"  FAILED for {cell_label} ({elapsed:.0f}s)")
            print(result.stdout[-500:])
            print(result.stderr[-500:])
            continue

        # Find the SPSA output dir created under cell_dir
        candidates = sorted(d for d in cell_dir.iterdir()
                           if d.is_dir() and d.name.startswith("2026-"))
        if not candidates:
            print(f"  no SPSA dir found in {cell_dir}")
            continue
        spsa_dir = candidates[-1]

        history = json.load(open(spsa_dir / "history.json"))
        final_beta = json.load(open(spsa_dir / "final_beta.json"))
        last_20_mean = (
            sum(h["outcome_mean"] for h in history[-20:]) / 20
            if len(history) >= 20 else float("nan")
        )
        rows.append({
            "noise": noise,
            "delay": delay,
            "cell": cell_label,
            "spsa_run": spsa_dir.name,
            "first_outcome": history[0]["outcome_mean"],
            "last_outcome": history[-1]["outcome_mean"],
            "last_20_outcome_mean": last_20_mean,
            "elapsed_sec": elapsed,
        })
        print(f"  {history[0]['outcome_mean']:.3f}m → {history[-1]['outcome_mean']:.3f}m"
              f"  (last-20 mean = {last_20_mean:.3f}m, elapsed = {elapsed:.0f}s)")

    summary_csv = out_root / "summary.csv"
    with open(summary_csv, "w", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow(r)
    print(f"\nSummary: {summary_csv}")


if __name__ == "__main__":
    main()
