"""Phase 2 closed-loop evaluation CLI.

Compares estimators under a fixed heuristic reach controller across a
focused (noise, delay) grid. For each cell rolls
``episodes_per_cell`` targets through ``run_closed_loop_episode`` and
writes per-episode metrics + grouped summary.

Usage::

    uv run python scripts/evaluate_closed_loop.py \\
        --config configs/closed_loop/heuristic_mvp.yaml
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from myoarm_fse.controllers import (
    BCController,
    EndpointErrorFeedbackController,
    JointSpacePDController,
    load_bc_policy,
    make_controller,
)
from myoarm_fse.data.rollout import EpisodeSpec
from myoarm_fse.envs.actions import ActionAdapter, detect_action_dim
from myoarm_fse.envs.factory import make_env
from myoarm_fse.envs.ik import (
    actuator_moment_dense,
    solve_ik,
    tip_jacobian_dense,
)
from myoarm_fse.envs.noise import SignalDependentMotorNoise
from myoarm_fse.envs.state import StateSpec
from myoarm_fse.envs.targets import TargetSet
from myoarm_fse.envs.wrappers import (
    DelayedObservationWrapper,
    NoisyObservationWrapper,
)
from myoarm_fse.estimators import (
    FixedGainKalmanEstimator,
    LearnedGainKalmanEstimator,
    StateAwareGainPredictor,
    StateAwareLearnedGainKalmanEstimator,
    load_learned_gain_model,
)
from myoarm_fse.evaluation import run_closed_loop_episode
from myoarm_fse.metrics import closed_loop_episode_summary
from myoarm_fse.models import load_model


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 2 closed-loop evaluation."
    )
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output-root", type=str, default=None)
    p.add_argument("--master-seed", type=int, default=None)
    p.add_argument("--smoke", action="store_true",
                   help="Run a single-cell single-episode smoke and exit.")
    return p.parse_args(argv)


def _load_config(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def _hash_config(cfg: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(cfg, sort_keys=True, default=str).encode()
    ).hexdigest()[:12]


def _make_eval_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


def _state_spec_from_model_config(model_config: dict[str, Any]) -> StateSpec:
    state_dim = int(model_config["architecture"]["state_dim"])
    if state_dim == 83:
        return StateSpec(qpos_dim=20, qvel_dim=20, act_dim=34)
    raise NotImplementedError(
        f"state_dim={state_dim} layout not auto-derivable"
    )


def _build_estimator(
    estimator_spec: dict[str, Any],
    *,
    forward_model: Any,
    state_spec: StateSpec,
    delay_steps: int,
    controller_name_for_learned: str,
    noise_sigma: dict[str, float],
    learned_models: dict[str, tuple[Any, dict[str, Any]]],
) -> Any:
    kind = estimator_spec["kind"]
    if kind == "fixed":
        return FixedGainKalmanEstimator(
            forward_model=forward_model,
            gain=float(estimator_spec["gain"]),
            state_spec=state_spec,
            delay_steps=int(delay_steps),
        )
    if kind == "learned":
        path = str(estimator_spec["learned_gain_model"])
        predictor, learned_cfg = learned_models[path]
        enc = learned_cfg.get("input_encoding", {})
        controller_names = tuple(
            enc.get("controller_names", ("random", "lowamp", "hold"))
        )
        sigma_field_order = tuple(
            enc.get("sigma_field_order", ("qpos", "qvel", "tip_pos", "reach_err"))
        )
        delay_max = int(enc.get("delay_max", 36))
        if isinstance(predictor, StateAwareGainPredictor):
            return StateAwareLearnedGainKalmanEstimator(
                forward_model=forward_model,
                gain_predictor=predictor,
                state_spec=state_spec,
                delay_steps=int(delay_steps),
                controller_name=controller_name_for_learned,
                noise_sigma=noise_sigma,
                controller_names=controller_names,
                sigma_field_order=sigma_field_order,
                delay_max=delay_max,
            )
        return LearnedGainKalmanEstimator(
            forward_model=forward_model,
            gain_predictor=predictor,
            state_spec=state_spec,
            delay_steps=int(delay_steps),
            controller_name=controller_name_for_learned,
            noise_sigma=noise_sigma,
            controller_names=controller_names,
            sigma_field_order=sigma_field_order,
            delay_max=delay_max,
        )
    raise ValueError(f"unknown estimator kind {kind!r}")


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    cfg = _load_config(args.config)
    if args.master_seed is not None:
        cfg["seed"] = int(args.master_seed)
    if args.output_root is not None:
        cfg["output_root"] = str(args.output_root)

    seed_master = int(cfg.get("seed", 0))
    env_id = str(cfg["env_id"])
    horizon = int(cfg["horizon"])

    print(f"Loaded config: {args.config}")
    print(f"  env_id: {env_id}  horizon: {horizon}")
    print(f"  forward_model: {cfg['forward_model']}")

    forward_model, model_config, _ = load_model(cfg["forward_model"])
    state_spec = _state_spec_from_model_config(model_config)
    state_dim = state_spec.dim
    forward_model.eval()

    # Pre-load every learned-gain model referenced by an estimator spec.
    learned_models: dict[str, tuple[Any, dict[str, Any]]] = {}
    for est in cfg["estimators"]:
        if est["kind"] == "learned":
            path = str(est["learned_gain_model"])
            if path not in learned_models:
                predictor, lconfig, _lm = load_learned_gain_model(path)
                learned_models[path] = (predictor, lconfig)
                print(f"  loaded learned-gain model: {path} "
                      f"(kind={lconfig.get('architecture', {}).get('kind')})")

    # Build env once and reuse across cells / episodes — target injection
    # happens per-episode via _inject_target.
    env = make_env(env_id, horizon=horizon)
    try:
        action_dim = detect_action_dim(env)
        action_adapter = ActionAdapter(action_dim=action_dim)

        target_set = TargetSet.load(cfg["target_set"])
        n_targets_available = target_set.n
        episodes_per_cell = int(cfg["episodes_per_cell"])
        if episodes_per_cell > n_targets_available:
            raise SystemExit(
                f"episodes_per_cell={episodes_per_cell} exceeds target_set.n="
                f"{n_targets_available}"
            )

        noise_conditions = cfg["noise_conditions"]
        delay_grid = [int(d) for d in cfg["delay_grid"]]
        estimators = list(cfg["estimators"])
        controller_spec = dict(cfg["controller"])
        bc_policy_cache = None
        bc_state_mask: np.ndarray | None = None
        if controller_spec.get("name") == "bc":
            bc_path = controller_spec.get("bc_policy_path")
            if not bc_path:
                raise SystemExit(
                    "controller.bc_policy_path required for name='bc'"
                )
            bc_policy_cache, bc_config, _bc_metrics = load_bc_policy(bc_path)
            mask_list = bc_config.get("architecture", {}).get(
                "state_feature_indices"
            )
            if mask_list is not None:
                mask_arr = np.asarray(mask_list, dtype=np.int64)
                if mask_arr.shape[0] != bc_policy_cache.state_dim:
                    raise SystemExit(
                        f"BC config state_feature_indices length "
                        f"{mask_arr.shape[0]} != policy.state_dim "
                        f"{bc_policy_cache.state_dim}"
                    )
                # Treat identity mask as None for cleaner logging.
                if mask_arr.shape[0] != bc_config.get(
                    "architecture", {}
                ).get("full_state_dim", state_dim):
                    bc_state_mask = mask_arr
            print(f"  loaded BC policy: {bc_path}  "
                  f"(state_dim={bc_policy_cache.state_dim}, "
                  f"action_dim={bc_policy_cache.action_dim}, "
                  f"mask={'yes' if bc_state_mask is not None else 'no'})")
        sdn_sigma_cfg = float(cfg.get("sdn", {}).get("sigma", 0.0))
        success_thresholds = tuple(
            float(t) for t in cfg.get("success_thresholds", (0.05,))
        )
        success_duration = int(cfg.get("success_duration", 10))
        skip_cold_start_use_delay = bool(
            cfg.get("skip_cold_start_steps_use_delay", True)
        )
        save_per_episode = bool(cfg.get("save_per_episode", False))

        n_cells = (
            len(estimators) * len(noise_conditions) * len(delay_grid)
            * episodes_per_cell
        )
        print(f"  grid: {len(estimators)} estimators x "
              f"{len(noise_conditions)} noise x {len(delay_grid)} delay x "
              f"{episodes_per_cell} eps = {n_cells} rollouts")

        rng_seq = np.random.SeedSequence(seed_master)
        child_seeds = rng_seq.spawn(n_cells)
        seed_iter = iter(child_seeds)

        eval_id = _make_eval_id()
        out_dir = Path(cfg.get("output_root", "runs/closed_loop")) / eval_id
        per_ep_dir = out_dir / "per_episode"
        if save_per_episode:
            per_ep_dir.mkdir(parents=True, exist_ok=True)
        else:
            out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "config.json").write_text(
            json.dumps({
                "eval_id": eval_id,
                "config_hash": _hash_config(cfg),
                "config": cfg,
                "forward_model_run_id": Path(cfg["forward_model"]).name,
            }, indent=2)
        )

        rows: list[dict[str, Any]] = []
        # When picking a "controller name" for the learned estimator's
        # one-hot, we evaluate the heuristic controller — but the learned
        # gain expects training-time controllers (random/lowamp/hold).
        # MVP choice: feed "random" since that is the most common label
        # in the training oracle. Logged in metadata.
        learned_ctrl_label = str(
            cfg.get("learned_estimator_controller_label", "random")
        )

        if args.smoke:
            # Single-cell smoke: first estimator, first noise, first delay,
            # first episode.
            estimators = estimators[:1]
            noise_conditions = {list(noise_conditions.keys())[0]:
                                list(noise_conditions.values())[0]}
            delay_grid = delay_grid[:1]
            episodes_per_cell = 1

        for est_spec in estimators:
            for noise_name, sigma_dict in noise_conditions.items():
                for delay in delay_grid:
                    for ep_idx in range(episodes_per_cell):
                        child = next(seed_iter)
                        seeds = child.generate_state(4).astype(np.int64)
                        obs_noise_seed = int(seeds[0])
                        sdn_seed = int(seeds[1])
                        controller_seed = int(seeds[2])
                        # seeds[3] reserved for future use.

                        # Build per-cell components.
                        obs_noise = (
                            NoisyObservationWrapper(
                                spec=state_spec, sigma=sigma_dict,
                                rng=obs_noise_seed,
                            )
                            if any(v != 0.0 for v in sigma_dict.values())
                            else None
                        )
                        obs_delay = (
                            DelayedObservationWrapper(
                                spec=state_spec, delay_steps=int(delay),
                            )
                            if int(delay) > 0 else None
                        )
                        sdn = (
                            SignalDependentMotorNoise(
                                action_dim=action_dim,
                                sigma=sdn_sigma_cfg, rng=sdn_seed,
                            )
                            if sdn_sigma_cfg > 0 else None
                        )

                        estimator = _build_estimator(
                            est_spec,
                            forward_model=forward_model,
                            state_spec=state_spec,
                            delay_steps=int(delay),
                            controller_name_for_learned=learned_ctrl_label,
                            noise_sigma=sigma_dict,
                            learned_models=learned_models,
                        )

                        target_pos = target_set.target_pos[ep_idx]

                        if controller_spec.get("name") == "bc":
                            bc_policy = bc_policy_cache
                            controller = BCController(
                                policy=bc_policy, action_dim=action_dim,
                                state_feature_indices=bc_state_mask,
                            )
                            controller.reset(seed=controller_seed)
                            ik_info = None
                        elif controller_spec.get("name") == "joint_pd":
                            # Pre-solve IK to get target_qpos and snapshot
                            # the moment-arm matrix at the env's current
                            # (neutral, post-reset) pose. Both calls
                            # preserve env state by default.
                            import mujoco as _mj
                            env.reset()
                            _mj.mj_forward(env.unwrapped.mj_model,
                                           env.unwrapped.mj_data)
                            target_qpos, ik_info = solve_ik(
                                env, target_pos,
                                max_iter=int(controller_spec.get("ik_max_iter", 200)),
                                tol=float(controller_spec.get("ik_tol", 0.01)),
                                damping=float(controller_spec.get(
                                    "ik_damping", 0.1
                                )),
                            )
                            moment_arm = actuator_moment_dense(env)
                            controller = JointSpacePDController(
                                action_dim=action_dim,
                                target_qpos=target_qpos,
                                moment_arm=moment_arm,
                                Kp=float(controller_spec.get("Kp", 10.0)),
                                Kd=float(controller_spec.get("Kd", 1.0)),
                                action_scale=float(controller_spec.get(
                                    "action_scale", 0.1
                                )),
                            )
                            controller.reset(seed=controller_seed)
                        elif controller_spec.get("name") == "endpoint_feedback":
                            # Capture the tip Jacobian and the moment-arm
                            # matrix at the env's current (neutral, post-
                            # reset) pose. No IK pre-step is needed.
                            import mujoco as _mj
                            env.reset()
                            _mj.mj_forward(env.unwrapped.mj_model,
                                           env.unwrapped.mj_data)
                            jacobian = tip_jacobian_dense(env)
                            moment_arm = actuator_moment_dense(env)
                            controller = EndpointErrorFeedbackController(
                                action_dim=action_dim,
                                target_pos=target_pos,
                                jacobian=jacobian,
                                moment_arm=moment_arm,
                                Kp=float(controller_spec.get("Kp", 30.0)),
                                Kd=float(controller_spec.get("Kd", 3.0)),
                                action_scale=float(controller_spec.get(
                                    "action_scale", 5.0
                                )),
                            )
                            controller.reset(seed=controller_seed)
                            ik_info = None
                        else:
                            controller = make_controller(
                                controller_spec,
                                action_dim=action_dim,
                                seed=controller_seed,
                            )
                            controller.reset(seed=controller_seed)
                            ik_info = None
                        spec_obj = EpisodeSpec(
                            episode_id=ep_idx,
                            target_id=str(int(target_set.seeds[ep_idx])),
                            target_split=target_set.split,
                            target_seed=int(target_set.seeds[ep_idx]),
                            controller_name=controller_spec["name"],
                            controller_seed=controller_seed,
                            sdn_seed=sdn_seed,
                            obs_noise_seed=obs_noise_seed,
                            config_hash=_hash_config(cfg),
                            meta={"estimator_name": est_spec["name"]},
                        )

                        result = run_closed_loop_episode(
                            env, controller, estimator, target_pos,
                            state_spec=state_spec,
                            action_adapter=action_adapter,
                            sdn=sdn, obs_noise=obs_noise, obs_delay=obs_delay,
                            obs_compose=str(cfg.get(
                                "obs_compose", "noisy_then_delayed"
                            )),
                            max_steps=horizon, spec=spec_obj,
                        )

                        from myoarm_fse.estimators.fixed_kalman import (
                            _flatten_log_states,
                        )
                        x_true = _flatten_log_states(result.log)
                        skip_cs = int(delay) if skip_cold_start_use_delay else 0
                        summary = closed_loop_episode_summary(
                            result.log,
                            x_est=result.x_est, x_true=x_true,
                            state_spec=state_spec,
                            success_thresholds=success_thresholds,
                            success_duration=success_duration,
                            skip_cold_start_steps=skip_cs,
                        )

                        # learned K stats if the estimator tracks them.
                        if hasattr(estimator, "k_history") and getattr(
                            estimator, "k_history", []
                        ):
                            k_arr = np.asarray(estimator.k_history, dtype=np.float64)
                            summary["learned_k_mean"] = float(k_arr.mean())
                            summary["learned_k_std"] = float(k_arr.std())
                            summary["learned_k_min"] = float(k_arr.min())
                            summary["learned_k_max"] = float(k_arr.max())
                        elif hasattr(estimator, "learned_k"):
                            summary["learned_k_mean"] = float(estimator.learned_k)
                            summary["learned_k_std"] = 0.0
                            summary["learned_k_min"] = float(estimator.learned_k)
                            summary["learned_k_max"] = float(estimator.learned_k)
                        elif est_spec["kind"] == "fixed":
                            g = float(est_spec["gain"])
                            summary["learned_k_mean"] = g
                            summary["learned_k_std"] = 0.0
                            summary["learned_k_min"] = g
                            summary["learned_k_max"] = g

                        row = {
                            "estimator": est_spec["name"],
                            "controller": controller_spec["name"],
                            "noise_condition": noise_name,
                            "delay_steps": int(delay),
                            "episode_id": ep_idx,
                            "target_index": ep_idx,
                            "target_seed": int(target_set.seeds[ep_idx]),
                            **summary,
                            "model_run_id": Path(cfg["forward_model"]).name,
                            "learned_gain_model_id": (
                                Path(est_spec.get("learned_gain_model", "")).name
                                if est_spec["kind"] == "learned" else ""
                            ),
                        }
                        rows.append(row)

                        if save_per_episode:
                            per_ep_path = (
                                per_ep_dir / f"{est_spec['name']}_n-{noise_name}_"
                                f"d-{int(delay)}_e-{ep_idx}.json"
                            )
                            per_ep_path.write_text(json.dumps(row, indent=2))

                        print(
                            f"  {est_spec['name']:<28s} noise={noise_name:>6s}  "
                            f"delay={int(delay):>2}  ep={ep_idx:>2}  "
                            f"final_tip={summary['final_tip_error']:.4f}  "
                            f"min_tip={summary['min_tip_error']:.4f}  "
                            f"S005={int(summary['success_005'])}  "
                            f"tip_est={summary['tip_estimation_error_mean']:.4f}"
                        )
    finally:
        env.close()

    # --- CSV ---
    if not rows:
        raise SystemExit("no rollouts ran; check config")
    columns = [
        "estimator", "controller", "noise_condition", "delay_steps",
        "episode_id", "target_index", "target_seed",
        "n_steps", "final_tip_error", "min_tip_error", "max_tip_error",
        "overshoot", "effort_norm",
        *[k for k in rows[0].keys() if k.startswith("success_")],
        "tip_estimation_error_mean", "tip_estimation_error_final",
        "state_mse_mean",
        "learned_k_mean", "learned_k_std", "learned_k_min", "learned_k_max",
        "model_run_id", "learned_gain_model_id",
    ]
    metrics_csv = out_dir / "metrics.csv"
    with open(metrics_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in columns})

    # --- summary (grouped) ---
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (row["estimator"], row["noise_condition"], int(row["delay_steps"]))
        groups[key].append(row)

    summary: list[dict[str, Any]] = []
    metric_fields = [
        "final_tip_error", "min_tip_error", "max_tip_error", "overshoot",
        "effort_norm", "tip_estimation_error_mean",
        "tip_estimation_error_final", "state_mse_mean",
    ]
    success_fields = [k for k in rows[0].keys() if k.startswith("success_")]
    for (est_name, noise_name, delay), group in sorted(groups.items()):
        entry: dict[str, Any] = {
            "estimator": est_name,
            "noise_condition": noise_name,
            "delay_steps": int(delay),
            "n": len(group),
        }
        for field in metric_fields:
            vals = [float(g[field]) for g in group]
            entry[f"{field}_mean"] = float(np.mean(vals))
            entry[f"{field}_std"] = float(np.std(vals))
        for field in success_fields:
            entry[f"{field}_rate"] = float(np.mean([float(g[field]) for g in group]))
        summary.append(entry)
    (out_dir / "summary.json").write_text(json.dumps({
        "eval_id": eval_id,
        "config_hash": _hash_config(cfg),
        "groups": summary,
    }, indent=2))

    print(f"\nSaved: {metrics_csv}")
    print(f"       {out_dir / 'summary.json'}")
    return out_dir


if __name__ == "__main__":
    main()
