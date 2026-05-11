"""Train Phase 3.2 Stage B state-aware gain predictor.

Usage::

    uv run python scripts/train_stage_b.py \\
        --config configs/estimators/learned_gain_stage_b.yaml \\
        --per-step-data runs/per_step_oracle/{id}/per_step.npz
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from myoarm_fse.estimators import (
    StateAwareTrainConfig,
    make_learned_gain_model_id,
    save_learned_gain_model,
    train_state_aware_gain_predictor,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train Phase 3.2 Stage B state-aware gain predictor."
    )
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--per-step-data", type=Path, required=True)
    p.add_argument("--master-seed", type=int, default=None)
    p.add_argument("--output-root", type=str, default=None)
    return p.parse_args(argv)


def _load_config(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    cfg = _load_config(args.config)
    if args.master_seed is not None:
        cfg["seed"] = int(args.master_seed)
    if args.output_root is not None:
        cfg.setdefault("output", {})["output_root"] = str(args.output_root)

    print(f"Loaded config: {args.config}")
    print(f"  seed: {cfg.get('seed')}")
    print(f"  per-step data: {args.per_step_data}")

    data = np.load(args.per_step_data)
    cond_features = data["cond_features"]
    state_features = data["state_features"]
    k_target = data["k_target"]
    groups = {
        "condition": data["group_condition"],
        "noise": data["group_noise"],
        "delay": data["group_delay"],
        "controller": data["group_controller"],
    }
    n = cond_features.shape[0]
    print(f"Loaded {n} samples")
    print(f"  cond dim={cond_features.shape[1]}, state dim={state_features.shape[1]}")
    print(f"  K* hist: mean={k_target.mean():.4f} median={np.median(k_target):.4f}")
    print(f"  K* <0.1={np.mean(k_target<0.1):.3f} <0.5={np.mean(k_target<0.5):.3f} "
          f">=0.9={np.mean(k_target>=0.9):.3f}")

    train_cfg_raw = dict(cfg.get("train", {}))
    if "cv_strategies" in train_cfg_raw and isinstance(
        train_cfg_raw["cv_strategies"], list
    ):
        train_cfg_raw["cv_strategies"] = tuple(train_cfg_raw["cv_strategies"])
    train_cfg_raw["seed"] = int(cfg.get("seed", 0))
    train_config = StateAwareTrainConfig.from_dict(train_cfg_raw)

    encoding_cfg = cfg.get("input_encoding", {})
    controller_names = tuple(
        encoding_cfg.get("controller_names", ("random", "lowamp", "hold"))
    )
    sigma_field_order = tuple(
        encoding_cfg.get(
            "sigma_field_order", ("qpos", "qvel", "tip_pos", "reach_err")
        )
    )
    state_feature_fields = tuple(
        encoding_cfg.get(
            "state_feature_fields", ("qpos", "qvel", "tip_pos", "reach_err")
        )
    )
    delay_max = int(encoding_cfg.get("delay_max", 36))
    hidden_dims = tuple(
        int(h) for h in cfg.get("architecture", {}).get("hidden_dims", (64, 64))
    )

    print(f"  cv_strategies: {train_config.cv_strategies}")
    print(f"  hidden_dims: {hidden_dims}")

    final_model, metrics = train_state_aware_gain_predictor(
        cond_features=cond_features,
        state_features=state_features,
        k_target=k_target,
        groups=groups,
        config=train_config,
        n_controllers=len(controller_names),
        n_sigma_fields=len(sigma_field_order),
        n_state_features=len(state_feature_fields),
        hidden_dims=hidden_dims,
    )

    print("\nCV results (abs_error of K_pred vs K*):")
    for strategy_name, result in metrics["cv_results"].items():
        if result.get("n_folds", 0) == 0:
            print(f"  {strategy_name:<11s}  (no folds)")
            continue
        print(
            f"  {strategy_name:<11s} n_folds={result['n_folds']:>2}  "
            f"mean={result['abs_error_mean']:.4f}  "
            f"median={result['abs_error_median']:.4f}  "
            f"max={result['abs_error_max']:.4f}"
        )
    print(
        f"\nFinal on full N={metrics['n_samples']}: "
        f"abs_error mean={metrics['final_train_abs_error_mean']:.4f}, "
        f"max={metrics['final_train_abs_error_max']:.4f}, "
        f"K* mean={metrics['target_k_mean']:.4f} K_pred mean={metrics['predicted_k_mean']:.4f}"
    )

    out_root = Path(
        cfg.get("output", {}).get("output_root", "runs/learned_gain_models")
    )
    model_id = make_learned_gain_model_id()
    out_dir = out_root / model_id

    persisted_config = {
        "architecture": {
            "kind": "StateAwareGainPredictor",
            "n_controllers": len(controller_names),
            "n_sigma_fields": len(sigma_field_order),
            "n_state_features": len(state_feature_fields),
            "hidden_dims": list(hidden_dims),
            "controller_names": list(controller_names),
            "sigma_field_order": list(sigma_field_order),
            "state_feature_fields": list(state_feature_fields),
            "delay_max": delay_max,
        },
        "train": asdict(train_config),
        "input_encoding": {
            "controller_names": list(controller_names),
            "sigma_field_order": list(sigma_field_order),
            "state_feature_fields": list(state_feature_fields),
            "delay_max": delay_max,
        },
        "noise_conditions": cfg.get("noise_conditions", {}),
        "raw_config": cfg,
    }
    info = {
        "per_step_data": str(args.per_step_data),
        "n_samples": int(n),
    }
    save_learned_gain_model(
        final_model, persisted_config, metrics, path=out_dir, info=info,
    )
    print(f"Saved model directory: {out_dir}")
    return out_dir


if __name__ == "__main__":
    main()
