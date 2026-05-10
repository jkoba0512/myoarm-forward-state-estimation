"""Collect myoArm episodes via run_episode.

Loads a YAML config, builds the env / controller / SDN / observation
wrappers, walks the target set, and writes one npz per episode plus a
run-level ``index.json``.

Usage::

    uv run python scripts/collect_episodes.py --config configs/episodes/default.yaml

Limited CLI overrides for ad-hoc debugging::

    --n-episodes      override config.n_episodes
    --master-seed     override config.master_seed
    --output-root     override config.output_root
    --target-set      override config.target_set
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from myoarm_fse.controllers import make_controller
from myoarm_fse.data.logger import (
    IndexEntry,
    RunIndex,
    hash_config,
    make_run_id,
)
from myoarm_fse.data.rollout import EpisodeSpec, run_episode
from myoarm_fse.envs.actions import ActionAdapter, detect_action_dim
from myoarm_fse.envs.factory import make_env
from myoarm_fse.envs.noise import SignalDependentMotorNoise
from myoarm_fse.envs.state import StateSpec
from myoarm_fse.envs.targets import TargetSet
from myoarm_fse.envs.wrappers import (
    DelayedObservationWrapper,
    NoisyObservationWrapper,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect myoArm episodes (Step 5 logger)."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--n-episodes", type=int, default=None)
    parser.add_argument("--master-seed", type=int, default=None)
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument("--target-set", type=str, default=None)
    return parser.parse_args(argv)


def _load_config(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def _apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if args.n_episodes is not None:
        config["n_episodes"] = args.n_episodes
    if args.master_seed is not None:
        config["master_seed"] = args.master_seed
    if args.output_root is not None:
        config["output_root"] = args.output_root
    if args.target_set is not None:
        config["target_set"] = args.target_set
    return config


def _derive_child_seeds(
    master_seed: int, n_episodes: int
) -> tuple[list[int], list[int], list[int]]:
    """Return (controller_seeds, sdn_seeds, obs_noise_seeds) of length n_episodes."""
    ss = np.random.SeedSequence(master_seed)
    children = ss.spawn(3 * n_episodes)
    controller = [int(c.generate_state(1)[0]) for c in children[0::3]]
    sdn = [int(c.generate_state(1)[0]) for c in children[1::3]]
    obs_noise = [int(c.generate_state(1)[0]) for c in children[2::3]]
    return controller, sdn, obs_noise


def _build_sdn(config: dict[str, Any], action_dim: int, seed: int):
    spec = config.get("sdn", {}) or {}
    sigma = float(spec.get("sigma", 0.0))
    if sigma <= 0.0:
        return None
    return SignalDependentMotorNoise(action_dim=action_dim, sigma=sigma, rng=seed)


def _build_obs_noise(config: dict[str, Any], state_spec: StateSpec, seed: int):
    spec = config.get("obs_noise", {}) or {}
    sigma_dict = spec.get("sigma", {}) or {}
    if not sigma_dict or all(v == 0 for v in sigma_dict.values()):
        return None
    return NoisyObservationWrapper(spec=state_spec, sigma=sigma_dict, rng=seed)


def _build_obs_delay(config: dict[str, Any], state_spec: StateSpec):
    delay = int(config.get("obs_delay_steps", 0))
    if delay <= 0:
        return None
    return DelayedObservationWrapper(spec=state_spec, delay_steps=delay)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = _apply_overrides(_load_config(args.config), args)
    print(f"Loaded config: {args.config}")
    for key in (
        "env_id",
        "target_set",
        "master_seed",
        "max_steps",
        "n_episodes",
        "obs_delay_steps",
        "obs_compose",
        "output_root",
    ):
        print(f"  {key}: {config.get(key)}")

    target_set = TargetSet.load(config["target_set"])
    n_targets = target_set.n
    n_requested = config.get("n_episodes")
    n_episodes = n_targets if n_requested is None else min(int(n_requested), n_targets)

    env = make_env(config["env_id"], max_steps_horizon := config.get("max_steps", 600))
    try:
        action_dim = detect_action_dim(env)
        adapter = ActionAdapter(action_dim=action_dim)
        # Probe state dims from the live env (matches state_schema mismatch check).
        from myoarm_fse.envs.extractors import extract_state

        env.reset()
        live_state = extract_state(env)
        state_spec = live_state.spec()

        controller_seeds, sdn_seeds, obs_noise_seeds = _derive_child_seeds(
            int(config.get("master_seed", 0)), n_episodes
        )

        run_id = make_run_id()
        config_hash = hash_config(config)
        out_dir = Path(config.get("output_root", "runs/episodes")) / run_id
        out_dir.mkdir(parents=True, exist_ok=True)

        index = RunIndex(
            run_id=run_id,
            created_at=datetime.now(timezone.utc).isoformat(),
            config_hash=config_hash,
            config=config,
            target_set_path=str(config["target_set"]),
            episodes=[],
        )

        print(f"Run id: {run_id}")
        print(f"Output: {out_dir}")
        print(f"Target set: {config['target_set']} (n={n_targets}, taking {n_episodes})")

        for i in range(n_episodes):
            target_pos = target_set.target_pos[i]
            target_seed = int(target_set.seeds[i])
            target_id = f"{target_set.split}:{i}"

            controller = make_controller(
                config["controller"], action_dim, controller_seeds[i]
            )
            sdn = _build_sdn(config, action_dim, sdn_seeds[i])
            obs_noise = _build_obs_noise(config, state_spec, obs_noise_seeds[i])
            obs_delay = _build_obs_delay(config, state_spec)

            spec = EpisodeSpec(
                episode_id=i,
                target_id=target_id,
                target_split=target_set.split,
                target_seed=target_seed,
                controller_name=type(controller).__name__,
                controller_seed=controller_seeds[i],
                sdn_seed=sdn_seeds[i],
                obs_noise_seed=obs_noise_seeds[i],
                config_hash=config_hash,
                meta={},
            )

            log = run_episode(
                env,
                controller,
                target_pos,
                state_spec=state_spec,
                action_adapter=adapter,
                sdn=sdn,
                obs_noise=obs_noise,
                obs_delay=obs_delay,
                obs_compose=str(config.get("obs_compose", "noisy_then_delayed")),
                max_steps=int(config.get("max_steps", 600)),
                spec=spec,
            )

            file_name = f"{i:04d}.npz"
            log.save(out_dir / file_name)
            index.append(
                IndexEntry(
                    episode_id=i,
                    file=file_name,
                    target_id=target_id,
                    target_seed=target_seed,
                    n_steps=log.n_steps,
                )
            )
            print(
                f"  [{i+1}/{n_episodes}] saved {file_name} "
                f"(n_steps={log.n_steps}, "
                f"final_reach_err_norm={float(np.linalg.norm(log.true_reach_err[-1])):.3f})"
            )

        index.save(out_dir / "index.json")
        print(f"  saved index.json with {len(index.episodes)} episodes")
    finally:
        env.close()


if __name__ == "__main__":
    main()
