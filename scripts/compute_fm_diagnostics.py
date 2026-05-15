"""Compute forward-model rollout error and bias per (cell, model) combination.

For each (cell × forward-model), run a few K=0 closed-loop episodes to
capture true state trajectories + action sequences, then offline replay
the forward model from multiple checkpoint states and aggregate:

* fm_rollout_mse_h{1,10,50}      per-cell mean squared state error
* fm_tip_err_h{1,10,50}          per-cell mean tip-position error (m)
* fm_bias_norm                   per-cell ||E[predicted - true]||_2 over field stack
* fm_bias_field_*                per-field bias norm
* fm_tip_signed_bias             E[||tip_pred|| - ||tip_true||] (over-/under-estimation in tip space)

Output: runs/diagnostics/geometry/fm_diag.csv with one row per (cell, fm).

Usage::

    uv run python scripts/compute_fm_diagnostics.py \
        --forward-models 2026-05-11T11-47-34Z 2026-05-11T09-53-07Z 2026-05-10T07-25-23Z \
        --output runs/diagnostics/geometry/fm_diag.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import mujoco
import numpy as np
import torch

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
from myoarm_fse.estimators import FixedGainKalmanEstimator
from myoarm_fse.evaluation.closed_loop import run_closed_loop_episode
from myoarm_fse.models import load_model


NOISE_PRESETS: dict[str, dict[str, float]] = {
    "none":  {"qpos": 0.0,  "qvel": 0.0,  "tip_pos": 0.0,  "reach_err": 0.0},
    "high":  {"qpos": 0.02, "qvel": 0.02, "tip_pos": 0.01, "reach_err": 0.01},
    "xhigh": {"qpos": 0.08, "qvel": 0.08, "tip_pos": 0.04, "reach_err": 0.04},
}

CELLS = [(n, d) for d in (0, 18) for n in ("none", "high", "xhigh")]

STATE_DIMS = {
    "qpos":      (0, 20),
    "qvel":      (20, 40),
    "act":       (40, 74),
    "tip_pos":   (74, 77),
    "target_pos": (77, 80),
    "reach_err": (80, 83),
}

ROLLOUT_HORIZONS = (1, 10, 50)


def stack_state(qpos, qvel, act, tip_pos, target_pos, reach_err):
    return np.concatenate([qpos, qvel, act, tip_pos, target_pos, reach_err], axis=-1)


def rollout_forward_model(
    forward_model,
    state0: np.ndarray,
    actions: np.ndarray,
    horizons: tuple[int, ...] = ROLLOUT_HORIZONS,
) -> dict[int, np.ndarray]:
    """Roll the forward model from state0 applying `actions` step by step.

    Returns dict horizon -> predicted_state (state_dim,) at step `horizon`.
    Assumes actions has shape (max(horizons), action_dim).
    """
    x = torch.tensor(state0, dtype=torch.float32).unsqueeze(0)  # (1, S)
    H = max(horizons)
    preds = {}
    with torch.no_grad():
        for h in range(1, H + 1):
            u = torch.tensor(actions[h - 1], dtype=torch.float32).unsqueeze(0)
            delta = forward_model(x, u)
            x = x + delta
            if h in horizons:
                preds[h] = x.squeeze(0).numpy()
    return preds


def run_cell_episode(
    env,
    forward_model,
    state_spec: StateSpec,
    target_pos: np.ndarray,
    noise: str,
    delay: int,
    seed: int,
):
    """Run one K=0 closed-loop episode and return per-step true state +
    api actions for offline forward-model rollout.
    """
    action_dim = detect_action_dim(env)
    action_adapter = ActionAdapter(action_dim=action_dim)

    env.reset(seed=seed)
    mujoco.mj_forward(env.unwrapped.mj_model, env.unwrapped.mj_data)
    target_qpos, _ = solve_ik(env, target_pos)
    moment_arm = actuator_moment_dense(env)
    controller = JointSpacePDController(
        action_dim=action_dim, target_qpos=target_qpos,
        moment_arm=moment_arm, Kp=30, Kd=3, action_scale=5,
    )
    controller.reset(seed=seed)

    sigma_dict = NOISE_PRESETS[noise]
    obs_noise = (
        NoisyObservationWrapper(spec=state_spec, sigma=sigma_dict, rng=seed)
        if any(v != 0 for v in sigma_dict.values()) else None
    )
    obs_delay = (
        DelayedObservationWrapper(spec=state_spec, delay_steps=delay)
        if delay > 0 else None
    )

    estimator = FixedGainKalmanEstimator(
        forward_model=forward_model, state_spec=state_spec,
        gain=0.0, delay_steps=delay,
    )

    result = run_closed_loop_episode(
        env, controller, estimator, target_pos,
        state_spec=state_spec, action_adapter=action_adapter,
        obs_noise=obs_noise, obs_delay=obs_delay,
        obs_compose="noisy_then_delayed", max_steps=600,
    )
    log = result.log
    true_state = stack_state(
        np.asarray(log.true_qpos), np.asarray(log.true_qvel),
        np.asarray(log.true_act), np.asarray(log.true_tip_pos),
        np.asarray(log.true_target_pos), np.asarray(log.true_reach_err),
    )
    actions = np.asarray(log.api_action)
    return true_state, actions


def aggregate_fm_diagnostics(
    forward_model,
    state_spec: StateSpec,
    target_pos: np.ndarray,
    env_id: str,
    noise: str,
    delay: int,
    n_episodes: int = 5,
    checkpoint_steps: tuple[int, ...] = (50, 150, 250, 350, 450),
) -> dict:
    """Per-cell aggregated forward-model diagnostics.

    Run `n_episodes` K=0 episodes, at each checkpoint step roll the
    forward model 50 ahead, aggregate.
    """
    h_errors = {h: [] for h in ROLLOUT_HORIZONS}
    h_tip_errors = {h: [] for h in ROLLOUT_HORIZONS}
    bias_full_list = []
    tip_signed_bias_list = []
    bias_per_field = {f: [] for f in STATE_DIMS}

    for ep in range(n_episodes):
        env = make_env(env_id, horizon=600)
        try:
            true_state, actions = run_cell_episode(
                env, forward_model, state_spec, target_pos,
                noise, delay, seed=ep,
            )
            T = true_state.shape[0]
            for s in checkpoint_steps:
                if s + max(ROLLOUT_HORIZONS) >= T:
                    continue
                preds = rollout_forward_model(
                    forward_model, true_state[s], actions[s:s + max(ROLLOUT_HORIZONS)],
                )
                for h in ROLLOUT_HORIZONS:
                    x_true = true_state[s + h]
                    x_pred = preds[h]
                    err = x_pred - x_true
                    h_errors[h].append(np.linalg.norm(err) ** 2)
                    tip_true = x_true[74:77]
                    tip_pred = x_pred[74:77]
                    h_tip_errors[h].append(np.linalg.norm(tip_pred - tip_true))
                # bias and per-field bias use h=10 prediction as the
                # representative rollout depth.
                h_ref = 10
                err_ref = preds[h_ref] - true_state[s + h_ref]
                bias_full_list.append(err_ref)
                for f, (lo, hi) in STATE_DIMS.items():
                    bias_per_field[f].append(np.linalg.norm(err_ref[lo:hi]))
                tip_signed_bias_list.append(
                    np.linalg.norm(preds[h_ref][74:77])
                    - np.linalg.norm(true_state[s + h_ref][74:77])
                )
        finally:
            env.close()

    out = {
        "fm_rollout_mse_h1": float(np.mean(h_errors[1])),
        "fm_rollout_mse_h10": float(np.mean(h_errors[10])),
        "fm_rollout_mse_h50": float(np.mean(h_errors[50])),
        "fm_tip_err_h1": float(np.mean(h_tip_errors[1])),
        "fm_tip_err_h10": float(np.mean(h_tip_errors[10])),
        "fm_tip_err_h50": float(np.mean(h_tip_errors[50])),
        "fm_bias_norm": float(np.linalg.norm(np.mean(bias_full_list, axis=0))),
        "fm_tip_signed_bias": float(np.mean(tip_signed_bias_list)),
    }
    for f in STATE_DIMS:
        out[f"fm_bias_{f}"] = float(np.mean(bias_per_field[f]))
    return out


def main(argv: list[str] | None = None) -> Path:
    p = argparse.ArgumentParser(description="Compute forward-model diagnostics.")
    p.add_argument("--forward-models", nargs="+",
                   default=["2026-05-11T11-47-34Z",
                            "2026-05-11T09-53-07Z",
                            "2026-05-10T07-25-23Z"],
                   help="Model run ids under runs/models/")
    p.add_argument("--target-index", type=int, default=0)
    p.add_argument("--target-set", default="runs/targets/train.npz")
    p.add_argument("--env-id", default="myoArmReachFixed-v0")
    p.add_argument("--n-episodes", type=int, default=3)
    p.add_argument("--output", type=Path,
                   default="runs/diagnostics/geometry/fm_diag.csv")
    args = p.parse_args(argv)

    state_spec = StateSpec(qpos_dim=20, qvel_dim=20, act_dim=34)
    target_set = TargetSet.load(args.target_set)
    target_pos = target_set.target_pos[args.target_index].astype(np.float32)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for model_id in args.forward_models:
        print(f"\n=== forward model {model_id} ===")
        fm, _, _ = load_model(f"runs/models/{model_id}")
        fm.eval()
        for noise, delay in CELLS:
            print(f"  cell (noise={noise}, delay={delay}) ... ", end="", flush=True)
            diag = aggregate_fm_diagnostics(
                fm, state_spec, target_pos, args.env_id,
                noise, delay, n_episodes=args.n_episodes,
            )
            diag["model_id"] = model_id
            diag["noise"] = noise
            diag["delay"] = delay
            rows.append(diag)
            print(f"h10_mse={diag['fm_rollout_mse_h10']:.3f} "
                  f"bias_norm={diag['fm_bias_norm']:.3f}")

    fieldnames = list(rows[0].keys())
    with open(args.output, "w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"\nSaved {args.output}")
    return args.output


if __name__ == "__main__":
    main()
