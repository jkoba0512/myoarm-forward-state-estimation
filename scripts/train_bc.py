"""Train a behavioral-cloning policy from ScriptedReach demonstrations."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from myoarm_fse.controllers import (
    BCTrainConfig,
    make_bc_model_id,
    save_bc_policy,
    train_bc_policy,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train BC policy on demos.")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--demos", type=Path, required=True,
                   help="Path to demos.npz (states + actions arrays)")
    p.add_argument("--output-root", type=str, default=None)
    p.add_argument("--master-seed", type=int, default=None)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.master_seed is not None:
        cfg["seed"] = int(args.master_seed)
    if args.output_root is not None:
        cfg.setdefault("output", {})["output_root"] = str(args.output_root)

    data = np.load(args.demos)
    states = data["states"]
    actions = data["actions"]
    print(f"Loaded demos from {args.demos}")
    print(f"  states: {states.shape}, actions: {actions.shape}")

    train_cfg_raw = dict(cfg.get("train", {}))
    train_cfg_raw["seed"] = int(cfg.get("seed", 0))
    train_config = BCTrainConfig.from_dict(train_cfg_raw)
    hidden_dims = tuple(
        int(h) for h in cfg.get("architecture", {}).get("hidden_dims", (256, 256))
    )
    print(f"  hidden_dims: {hidden_dims}, epochs: {train_config.epochs}, "
          f"batch: {train_config.batch_size}")

    model, metrics = train_bc_policy(
        states=states, actions=actions, config=train_config, hidden_dims=hidden_dims,
    )

    print("\nLoss curve (epoch=0, mid, last):")
    h = metrics["history"]
    if len(h) >= 3:
        for idx in (0, len(h) // 2, len(h) - 1):
            print(f"  epoch {h[idx]['epoch']:>3d}: train={h[idx]['train_loss']:.5f}  "
                  f"val={h[idx]['val_loss']:.5f}")
    print(f"\nFinal train loss: {metrics['final_train_loss']:.5f}")
    print(f"Final val loss:   {metrics['final_val_loss']:.5f}")

    out_root = Path(
        cfg.get("output", {}).get("output_root", "runs/bc_policies")
    )
    model_id = make_bc_model_id()
    out_dir = out_root / model_id
    persisted_config = {
        "architecture": {
            "state_dim": int(states.shape[1]),
            "action_dim": int(actions.shape[1]),
            "hidden_dims": list(hidden_dims),
        },
        "train": asdict(train_config),
        "raw_config": cfg,
        "demos_path": str(args.demos),
    }
    save_bc_policy(model, persisted_config, metrics, path=out_dir)
    print(f"\nSaved: {out_dir}")
    return out_dir


if __name__ == "__main__":
    main()
