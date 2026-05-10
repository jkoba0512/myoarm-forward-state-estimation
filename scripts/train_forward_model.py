"""Train a ``ForwardMLP`` on one or more ``TransitionDataset`` files.

Usage::

    uv run python scripts/train_forward_model.py --config configs/models/mlp.yaml

Limited CLI overrides for ad-hoc experiments::

    --master-seed     override config.seed
    --epochs          override config.train.epochs
    --output-root     override config.output.output_root
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from myoarm_fse.models import (
    ForwardMLP,
    TrainConfig,
    TransitionDataset,
    load_model,  # noqa: F401  -- exposed for ad-hoc REPL use
    make_model_id,
    make_train_val_split,
    rollout_predictions,
    save_model,
    setup_seeds,
    train_forward_model,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a ForwardMLP forward model.")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--master-seed", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--output-root", type=str, default=None)
    return p.parse_args(argv)


def _load_config(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def _apply_overrides(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if args.master_seed is not None:
        cfg["seed"] = int(args.master_seed)
    if args.epochs is not None:
        cfg.setdefault("train", {})["epochs"] = int(args.epochs)
    if args.output_root is not None:
        cfg.setdefault("output", {})["output_root"] = str(args.output_root)
    return cfg


def _load_concat_dataset(paths: list[str]) -> TransitionDataset:
    if not paths:
        raise ValueError("data.dataset_paths must contain at least one path")
    parts = [TransitionDataset.load(p) for p in paths]
    if len(parts) == 1:
        return parts[0]
    state_dim = parts[0].state_dim
    action_dim = parts[0].action_dim
    for ds in parts[1:]:
        if ds.state_dim != state_dim or ds.action_dim != action_dim:
            raise ValueError(
                "all datasets must share state_dim/action_dim "
                f"({state_dim}, {action_dim})"
            )
    # Reindex episode_index to be unique across the concatenation.  Source
    # runs each number their own episodes from 0, so concatenating without
    # rewriting episode_id would make split_by_episode collapse colliding
    # ids (e.g. all episodes named "0") into a single split.  Preserve the
    # source id under "source_episode_id" and reassign episode_id to a
    # global sequential index.
    new_episode_indexes: list[np.ndarray] = []
    new_metadata: list[dict[str, Any]] = []
    offset = 0
    for ds in parts:
        new_episode_indexes.append(ds.episode_index + offset)
        for local_idx, meta in enumerate(ds.episode_metadata):
            entry = dict(meta)
            entry["source_episode_id"] = entry.get("episode_id")
            entry["episode_id"] = offset + local_idx
            new_metadata.append(entry)
        offset += ds.n_episodes
    return TransitionDataset(
        x=np.concatenate([ds.x for ds in parts], axis=0),
        u=np.concatenate([ds.u for ds in parts], axis=0),
        x_next=np.concatenate([ds.x_next for ds in parts], axis=0),
        dx=np.concatenate([ds.dx for ds in parts], axis=0),
        episode_index=np.concatenate(new_episode_indexes, axis=0),
        state_dim=state_dim,
        action_dim=action_dim,
        n_episodes=offset,
        episode_metadata=tuple(new_metadata),
    )


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    cfg = _apply_overrides(_load_config(args.config), args)

    print(f"Loaded config: {args.config}")
    print(f"  seed: {cfg.get('seed')}")
    print(f"  hidden_dims: {cfg.get('architecture', {}).get('hidden_dims')}")
    print(f"  data.dataset_paths: {cfg.get('data', {}).get('dataset_paths')}")
    print(f"  output.output_root: {cfg.get('output', {}).get('output_root')}")

    dataset = _load_concat_dataset(list(cfg["data"]["dataset_paths"]))
    print(
        f"Dataset: N={dataset.n}, n_episodes={dataset.n_episodes}, "
        f"state_dim={dataset.state_dim}, action_dim={dataset.action_dim}"
    )

    train_cfg_dict = dict(cfg.get("train", {}))
    train_config = TrainConfig.from_dict(train_cfg_dict | {"seed": int(cfg["seed"])})
    train_ds, val_ds = make_train_val_split(dataset, val_step=train_config.val_step)
    print(
        f"Split: train N={train_ds.n} ({train_ds.n_episodes} ep) / "
        f"val N={val_ds.n} ({val_ds.n_episodes} ep)"
    )

    seeds = setup_seeds(train_config.seed)

    model = ForwardMLP(
        state_dim=dataset.state_dim,
        action_dim=dataset.action_dim,
        hidden_dims=tuple(int(h) for h in cfg["architecture"]["hidden_dims"]),
    )
    print(f"Model: {model} (params={model.num_parameters()})")

    model, metrics = train_forward_model(
        model, train_ds, val_ds, train_config, seeds=seeds,
    )
    print(
        f"Trained: best_epoch={metrics['best_epoch']}, "
        f"best_val_loss={metrics['best_val_loss']:.6f}, "
        f"epochs_run={len(metrics['train_loss_history'])}"
    )

    horizons = tuple(int(h) for h in cfg.get("eval", {}).get("rollout_horizons", (1, 10, 50)))
    rollout_metrics = rollout_predictions(model, val_ds, horizons=horizons)
    metrics["rollout_metrics"] = {
        str(h): r for h, r in rollout_metrics.items()
    }
    for h, r in rollout_metrics.items():
        print(
            f"  rollout h={h}: mse={r['rollout_mse']:.6f}, "
            f"tip_err={r['tip_prediction_error']:.6f}"
        )

    out_root = Path(cfg.get("output", {}).get("output_root", "runs/models"))
    model_id = make_model_id()
    out_dir = out_root / model_id

    persisted_config = {
        "architecture": {
            "kind": "ForwardMLP",
            "state_dim": dataset.state_dim,
            "action_dim": dataset.action_dim,
            "hidden_dims": list(model.hidden_dims),
            "activation": "relu",
            "norm": "layernorm",
        },
        "train": asdict(train_config),
        "data": {
            "dataset_paths": list(cfg["data"]["dataset_paths"]),
            "n_transitions": int(dataset.n),
            "state_dim": dataset.state_dim,
            "action_dim": dataset.action_dim,
            "n_episodes": dataset.n_episodes,
        },
        "eval": {"rollout_horizons": list(horizons)},
        "raw_config": cfg,
    }
    save_model(model, persisted_config, metrics, path=out_dir)
    print(f"Saved model directory: {out_dir}")
    return out_dir


if __name__ == "__main__":
    main()
