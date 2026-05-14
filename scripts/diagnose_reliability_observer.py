"""Diagnostic dump of K_t, reliability_t, and innovation_t for the
reliability-adaptive observer on a single representative episode.

Used by ``scripts/make_paper_figures.py`` to render the K_t trajectory,
reliability-vs-noise condition figures, and field-wise correction-gain
specialization bar charts described in §4 of the paper. The diagnostic
loads a trained β configuration (either SPSA-learned or default) and
runs one full episode at the requested (noise, delay) cell.

Usage::

    uv run python scripts/diagnose_reliability_observer.py \
        --beta-config runs/reliability_adaptive_v2/<id>/final_beta.json \
        --noise none --delay 18 \
        --target-index 0 \
        --output runs/diagnostics/<name>.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from myoarm_fse.controllers import JointSpacePDController
from myoarm_fse.envs.actions import ActionAdapter, detect_action_dim
from myoarm_fse.envs.factory import make_env
from myoarm_fse.envs.ik import actuator_moment_dense, solve_ik
from myoarm_fse.envs.state import StateSpec
from myoarm_fse.envs.targets import TargetSet
from myoarm_fse.envs.wrappers import (
    DelayedObservationWrapper,
    NoisyObservationWrapper,
)
from myoarm_fse.estimators import (
    ReliabilityAdaptiveConfig,
    ReliabilityAdaptiveObserver,
)
from myoarm_fse.evaluation.closed_loop import run_closed_loop_episode
from myoarm_fse.metrics.reaching import minimum_tip_error
from myoarm_fse.models import load_model


_NOISE_PRESETS: dict[str, dict[str, float]] = {
    "none":  {"qpos": 0.0,  "qvel": 0.0,  "tip_pos": 0.0,  "reach_err": 0.0},
    "high":  {"qpos": 0.02, "qvel": 0.02, "tip_pos": 0.01, "reach_err": 0.01},
    "xhigh": {"qpos": 0.08, "qvel": 0.08, "tip_pos": 0.04, "reach_err": 0.04},
}


def main(argv: list[str] | None = None) -> Path:
    p = argparse.ArgumentParser(description="Diagnose reliability observer state.")
    p.add_argument("--forward-model",
                   default="runs/models/2026-05-11T11-47-34Z")
    p.add_argument("--beta-config", type=Path, default=None,
                   help="JSON with {beta0, beta1, alpha?, epsilon?, var_init?}. "
                        "Defaults to within-trial v1 (β₀=0, β₁=0.5).")
    p.add_argument("--noise", default="none", choices=list(_NOISE_PRESETS))
    p.add_argument("--delay", type=int, default=18)
    p.add_argument("--target-index", type=int, default=0)
    p.add_argument("--target-set", default="runs/targets/train.npz")
    p.add_argument("--env-id", default="myoArmReachFixed-v0")
    p.add_argument("--horizon", type=int, default=600)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)

    fm, _, _ = load_model(args.forward_model)
    fm.eval()
    state_spec = StateSpec(qpos_dim=20, qvel_dim=20, act_dim=34)
    target_set = TargetSet.load(args.target_set)
    target_pos = target_set.target_pos[args.target_index].astype(np.float32)

    # Build β config
    if args.beta_config is not None:
        with open(args.beta_config) as f:
            beta_dict = json.load(f)
        cfg = ReliabilityAdaptiveConfig(
            alpha=float(beta_dict.get("alpha", 0.05)),
            epsilon=float(beta_dict.get("epsilon", 1e-6)),
            var_init=float(beta_dict.get("var_init", 1.0)),
            beta0={k: float(v) for k, v in beta_dict["beta0"].items()},
            beta1={k: float(v) for k, v in beta_dict["beta1"].items()},
            target_pos_gain=float(beta_dict.get("target_pos_gain", 1.0)),
        )
    else:
        cfg = ReliabilityAdaptiveConfig()  # defaults

    env = make_env(args.env_id, horizon=args.horizon)
    try:
        action_dim = detect_action_dim(env)
        action_adapter = ActionAdapter(action_dim=action_dim)

        # Build joint-PD controller
        env.reset()
        mujoco.mj_forward(env.unwrapped.mj_model, env.unwrapped.mj_data)
        target_qpos, _ = solve_ik(env, target_pos)
        moment_arm = actuator_moment_dense(env)
        controller = JointSpacePDController(
            action_dim=action_dim, target_qpos=target_qpos,
            moment_arm=moment_arm, Kp=30, Kd=3, action_scale=5,
        )
        controller.reset(seed=0)

        sigma_dict = _NOISE_PRESETS[args.noise]
        obs_noise = (
            NoisyObservationWrapper(spec=state_spec, sigma=sigma_dict, rng=0)
            if any(v != 0 for v in sigma_dict.values()) else None
        )
        obs_delay = (
            DelayedObservationWrapper(spec=state_spec, delay_steps=args.delay)
            if args.delay > 0 else None
        )

        observer = ReliabilityAdaptiveObserver(
            forward_model=fm, state_spec=state_spec,
            delay_steps=args.delay, config=cfg,
        )

        result = run_closed_loop_episode(
            env, controller, observer, target_pos,
            state_spec=state_spec, action_adapter=action_adapter,
            obs_noise=obs_noise, obs_delay=obs_delay,
            obs_compose="noisy_then_delayed", max_steps=args.horizon,
        )

        # Collect per-step histories
        innovation = observer.innovation_history()  # (T, 83)
        k_hist = observer.field_k_history()          # dict[field, (T,)]
        rel_hist = observer.field_reliability_history()
        var_hist = observer.field_var_history()

        # Compute tip-to-target distance over time
        tip_pos = np.asarray(result.log.true_tip_pos)  # (T, 3)
        tgt_pos = np.asarray(result.log.true_target_pos)
        tip_to_target = np.linalg.norm(tip_pos - tgt_pos, axis=1)

        out_data = dict(
            innovation=innovation,
            tip_to_target=tip_to_target.astype(np.float32),
            min_tip=float(minimum_tip_error(result.log)),
            noise=args.noise,
            delay=args.delay,
            target_index=args.target_index,
        )
        for f in ("qpos", "qvel", "act", "tip_pos", "reach_err"):
            out_data[f"k_{f}"] = k_hist[f]
            out_data[f"reliability_{f}"] = rel_hist[f]
            out_data[f"var_{f}"] = var_hist[f]

        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.output, **out_data)
        print(f"Saved {args.output}")
        print(f"  min_tip = {out_data['min_tip']:.4f} m")
        print("  Final K per field:")
        for f in ("qpos", "qvel", "act", "tip_pos", "reach_err"):
            print(f"    {f:12s}: K={k_hist[f][-1]:.3f}  "
                  f"reliability={rel_hist[f][-1]:.2e}")
        return args.output
    finally:
        env.close()


if __name__ == "__main__":
    main()
