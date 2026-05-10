"""Train the Phase 3.2 Stage A condition-level gain predictor.

Usage::

    uv run python scripts/train_learned_gain.py \\
        --config configs/estimators/learned_gain_default.yaml \\
        --oracle-table runs/estimators/{eval_id}/best_by_condition.csv

The CLI loads the YAML config, attaches sigma vectors (from
``noise_conditions``) to each oracle row, runs the configured CV
strategies, and saves the final model directory under ``output_root``.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from myoarm_fse.estimators import (
    LearnedGainTrainConfig,
    load_oracle_table,
    make_learned_gain_model_id,
    save_learned_gain_model,
    train_gain_predictor,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train Phase 3.2 Stage A condition-level gain predictor."
    )
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--oracle-table", type=Path, required=True)
    p.add_argument("--master-seed", type=int, default=None)
    p.add_argument("--output-root", type=str, default=None)
    return p.parse_args(argv)


def _load_config(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def _apply_overrides(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if args.master_seed is not None:
        cfg["seed"] = int(args.master_seed)
    if args.output_root is not None:
        cfg.setdefault("output", {})["output_root"] = str(args.output_root)
    return cfg


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    cfg = _apply_overrides(_load_config(args.config), args)

    print(f"Loaded config: {args.config}")
    print(f"  seed: {cfg.get('seed')}")
    print(f"  hidden_dims: {cfg.get('architecture', {}).get('hidden_dims')}")
    print(f"  oracle_table: {args.oracle_table}")
    print(f"  output_root: {cfg.get('output', {}).get('output_root')}")

    oracle_rows = load_oracle_table(args.oracle_table)
    print(f"Loaded oracle table: {len(oracle_rows)} rows")

    run_to_controller = cfg.get("run_to_controller") or {}
    if run_to_controller:
        # best_by_condition.csv stores run_id (timestamp) as the controller
        # column. The encoder expects canonical names (random/lowamp/hold).
        # Apply the mapping in-place; rows whose controller is already a
        # canonical name (e.g. in tests) pass through unchanged.
        unmapped: set[str] = set()
        for row in oracle_rows:
            ctrl = row["controller"]
            if ctrl in run_to_controller:
                row["controller"] = run_to_controller[ctrl]
            elif ctrl not in set(run_to_controller.values()):
                unmapped.add(ctrl)
        if unmapped:
            raise SystemExit(
                f"oracle controllers not in run_to_controller and not "
                f"already canonical names: {sorted(unmapped)}. "
                f"Update config.run_to_controller."
            )
        print(
            f"  Applied run_to_controller map "
            f"({len(run_to_controller)} entries) to oracle rows."
        )

    train_cfg_raw = dict(cfg.get("train", {}))
    if "cv_strategies" in cfg:
        train_cfg_raw["cv_strategies"] = tuple(cfg["cv_strategies"])
    train_cfg_raw["seed"] = int(cfg.get("seed", 0))
    train_config = LearnedGainTrainConfig.from_dict(train_cfg_raw)

    encoding_cfg = cfg.get("input_encoding", {})
    controller_names = tuple(
        encoding_cfg.get("controller_names", ("random", "lowamp", "hold"))
    )
    sigma_field_order = tuple(
        encoding_cfg.get(
            "sigma_field_order",
            ("qpos", "qvel", "tip_pos", "reach_err"),
        )
    )
    delay_max = int(encoding_cfg.get("delay_max", 6))
    hidden_dims = tuple(
        int(h) for h in cfg.get("architecture", {}).get("hidden_dims", (32, 32))
    )
    noise_conditions = cfg.get("noise_conditions") or {}
    if not noise_conditions:
        raise SystemExit(
            "config.noise_conditions is required so the trainer can attach "
            "sigma vectors to each oracle row's noise_condition label."
        )

    print(f"  cv_strategies: {train_config.cv_strategies}")
    print(f"  controller_names: {controller_names}")
    print(f"  sigma_field_order: {sigma_field_order}")
    print(f"  delay_max: {delay_max}")
    print(f"  noise_conditions: {sorted(noise_conditions.keys())}")

    final_model, metrics = train_gain_predictor(
        oracle_rows,
        config=train_config,
        noise_conditions=noise_conditions,
        controller_names=controller_names,
        sigma_field_order=sigma_field_order,
        delay_max=delay_max,
        hidden_dims=hidden_dims,
    )

    print("\nCV results (abs_error of K_pred vs oracle K):")
    for strategy_name, result in metrics["cv_results"].items():
        print(
            f"  {strategy_name:<11s} n_folds={result['n_folds']:>2}  "
            f"mean={result['abs_error_mean']:.4f}  "
            f"median={result['abs_error_median']:.4f}  "
            f"max={result['abs_error_max']:.4f}"
        )
    print(
        f"\nFinal model on full N={metrics['n_oracle_rows']}: "
        f"abs_error mean={metrics['final_train_abs_error_mean']:.4f}, "
        f"max={metrics['final_train_abs_error_max']:.4f}"
    )

    out_root = Path(cfg.get("output", {}).get("output_root", "runs/learned_gain_models"))
    model_id = make_learned_gain_model_id()
    out_dir = out_root / model_id

    persisted_config = {
        "architecture": {
            "kind": "GainPredictor",
            "n_controllers": len(controller_names),
            "n_sigma_fields": len(sigma_field_order),
            "hidden_dims": list(hidden_dims),
            "controller_names": list(controller_names),
            "sigma_field_order": list(sigma_field_order),
            "delay_max": delay_max,
        },
        "train": asdict(train_config),
        "input_encoding": {
            "controller_names": list(controller_names),
            "sigma_field_order": list(sigma_field_order),
            "delay_max": delay_max,
        },
        "noise_conditions": noise_conditions,
        "raw_config": cfg,
    }
    info = {
        "oracle_table_path": str(args.oracle_table),
        "n_oracle_rows": len(oracle_rows),
    }
    save_learned_gain_model(
        final_model, persisted_config, metrics, path=out_dir, info=info,
    )
    print(f"Saved model directory: {out_dir}")
    return out_dir


if __name__ == "__main__":
    main()
