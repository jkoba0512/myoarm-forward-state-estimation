"""Closed-loop rollout helper for Phase 2.

The open-loop ``data.rollout.run_episode`` feeds the controller the
*observation* (post-noise/delay wrappers). Closed-loop instead routes
the observation through an ``Estimator`` first and feeds the estimator
output to the controller, so estimation quality directly affects task
performance.

This module is a thin wrapper around the same env / SDN / adapter
machinery used by ``run_episode``; we duplicate the per-step buffer
recording rather than coupling the two pipelines, since they diverge in
exactly the "controller input" position.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import mujoco
import numpy as np

from myoarm_fse.controllers.base import Controller
from myoarm_fse.data.rollout import EpisodeSpec, _apply_obs, _inject_target
from myoarm_fse.data.schema import EpisodeLog
from myoarm_fse.envs.actions import ActionAdapter
from myoarm_fse.envs.extractors import extract_ctrl, extract_state
from myoarm_fse.envs.noise import SignalDependentMotorNoise
from myoarm_fse.envs.state import MyoArmState, StateSpec
from myoarm_fse.envs.wrappers import (
    DelayedObservationWrapper,
    NoisyObservationWrapper,
)
from myoarm_fse.estimators.base import Estimator

_DT_F32 = np.dtype(np.float32)
_DT_INT = np.dtype(np.int64)
_DT_BOOL = np.dtype(np.bool_)
_CART_DIM: int = 3
_VALID_OBS_COMPOSE: tuple[str, ...] = ("noisy_then_delayed", "delayed_then_noisy")


@dataclass(frozen=True, eq=False)
class ClosedLoopEpisodeResult:
    """One episode of closed-loop rollout.

    ``log`` is the same shape as the open-loop ``EpisodeLog`` (true_*
    are oracle state, obs_* are the post-wrapper view the estimator
    saw, excitation_* are the action stream the env actually received).
    ``x_est`` is the estimator's output over the episode, aligned to
    ``log`` step-for-step.
    """

    log: EpisodeLog
    x_est: np.ndarray  # (T, state_dim)


def run_closed_loop_episode(
    env: Any,
    controller: Controller,
    estimator: Estimator,
    target_pos: np.ndarray,
    *,
    state_spec: StateSpec,
    action_adapter: ActionAdapter,
    sdn: SignalDependentMotorNoise | None = None,
    obs_noise: NoisyObservationWrapper | None = None,
    obs_delay: DelayedObservationWrapper | None = None,
    obs_compose: str = "noisy_then_delayed",
    max_steps: int = 600,
    spec: EpisodeSpec | None = None,
) -> ClosedLoopEpisodeResult:
    """Run one closed-loop episode with controller driven by ``estimator``.

    Per-step order:

    1. Read the env's true state.
    2. Apply observation wrappers (delay/noise) -> ``observed``.
    3. ``estimator.step(observed.flatten(), prev_u)`` -> ``x_est_flat``.
    4. Reconstruct ``MyoArmState`` from ``x_est_flat``.
    5. ``controller.act(x_est_state)`` -> excitation_command.
    6. SDN / clip -> excitation, then ``action_adapter`` -> api_action.
    7. ``env.step(api_action)``.
    8. Log everything.

    At ``t = 0`` the estimator's prediction step uses ``u_prev = 0`` (no
    action has been applied yet). The estimator's ``reset`` is called
    with the env's actual initial true state so the buffer is correctly
    seeded.
    """
    if obs_compose not in _VALID_OBS_COMPOSE:
        raise ValueError(
            f"obs_compose must be one of {_VALID_OBS_COMPOSE}, got {obs_compose!r}"
        )
    if max_steps < 0:
        raise ValueError(f"max_steps must be >= 0, got {max_steps}")
    if not isinstance(controller, Controller):
        raise TypeError(
            f"controller must implement the Controller protocol, "
            f"got {type(controller).__name__}"
        )
    if not isinstance(estimator, Estimator):
        raise TypeError(
            f"estimator must implement the Estimator protocol, "
            f"got {type(estimator).__name__}"
        )

    target_pos_arr = np.asarray(target_pos, dtype=_DT_F32).copy()
    if target_pos_arr.shape != (_CART_DIM,):
        raise ValueError(
            f"target_pos must have shape (3,), got {target_pos_arr.shape}"
        )

    if spec is None:
        spec = EpisodeSpec(
            episode_id=0, target_id="", target_split="", target_seed=0,
            controller_name=type(controller).__name__, controller_seed=0,
        )

    env.reset()
    _inject_target(env, target_pos_arr)
    initial_true_state = extract_state(env)
    if obs_delay is not None:
        obs_delay.reset(initial_true_state)
    estimator.reset(initial_true_state.flatten())

    qpos_dim = state_spec.qpos_dim
    qvel_dim = state_spec.qvel_dim
    act_dim = state_spec.act_dim
    state_dim = state_spec.dim
    action_dim = action_adapter.action_dim
    dt = float(env.unwrapped.dt)
    T = max_steps

    # Pre-allocate buffers, same shape as run_episode.
    step_buf = np.empty(T, dtype=_DT_INT)
    time_buf = np.empty(T, dtype=_DT_F32)
    true_qpos = np.empty((T, qpos_dim), dtype=_DT_F32)
    true_qvel = np.empty((T, qvel_dim), dtype=_DT_F32)
    true_act = np.empty((T, act_dim), dtype=_DT_F32)
    true_tip_pos = np.empty((T, _CART_DIM), dtype=_DT_F32)
    true_target_pos = np.empty((T, _CART_DIM), dtype=_DT_F32)
    true_reach_err = np.empty((T, _CART_DIM), dtype=_DT_F32)
    obs_qpos = np.empty((T, qpos_dim), dtype=_DT_F32)
    obs_qvel = np.empty((T, qvel_dim), dtype=_DT_F32)
    obs_act = np.empty((T, act_dim), dtype=_DT_F32)
    obs_tip_pos = np.empty((T, _CART_DIM), dtype=_DT_F32)
    obs_target_pos = np.empty((T, _CART_DIM), dtype=_DT_F32)
    obs_reach_err = np.empty((T, _CART_DIM), dtype=_DT_F32)
    neural_command_buf = np.empty((T, action_dim), dtype=_DT_F32)
    excitation_command_buf = np.empty((T, action_dim), dtype=_DT_F32)
    excitation_buf = np.empty((T, action_dim), dtype=_DT_F32)
    api_action_buf = np.empty((T, action_dim), dtype=_DT_F32)
    last_ctrl_buf = np.empty((T, action_dim), dtype=_DT_F32)
    reward_buf = np.empty(T, dtype=_DT_F32)
    terminated_buf = np.empty(T, dtype=_DT_BOOL)
    truncated_buf = np.empty(T, dtype=_DT_BOOL)
    x_est_buf = np.empty((T, state_dim), dtype=_DT_F32)

    n_steps = 0
    prev_u = np.zeros(action_dim, dtype=_DT_F32)

    for t in range(T):
        true_state = extract_state(env)
        observed = _apply_obs(true_state, obs_noise, obs_delay, obs_compose)

        x_est_flat = estimator.step(observed.flatten(), prev_u)
        if x_est_flat.shape != (state_dim,):
            raise ValueError(
                f"estimator returned shape {x_est_flat.shape}, expected ({state_dim},)"
            )
        x_est_state = MyoArmState.unflatten(x_est_flat, state_spec)

        neural_command = controller.act(x_est_state)
        excitation_command = np.asarray(neural_command, dtype=_DT_F32)
        if excitation_command.shape != (action_dim,):
            raise ValueError(
                f"controller returned shape {excitation_command.shape}, "
                f"expected ({action_dim},)"
            )
        if not np.isfinite(excitation_command).all():
            raise ValueError("controller produced non-finite excitation_command")
        if (excitation_command < 0.0).any() or (excitation_command > 1.0).any():
            raise ValueError("controller produced excitation_command outside [0, 1]")

        if sdn is not None:
            excitation = sdn.apply(excitation_command)
        else:
            excitation = action_adapter.clip_excitation(excitation_command)
        api_action = action_adapter.excitation_to_api_action(excitation)
        _, reward, terminated, truncated, _info = env.step(api_action)
        last_ctrl = extract_ctrl(env)

        step_buf[t] = t
        time_buf[t] = t * dt
        true_qpos[t] = true_state.qpos
        true_qvel[t] = true_state.qvel
        true_act[t] = true_state.act
        true_tip_pos[t] = true_state.tip_pos
        true_target_pos[t] = true_state.target_pos
        true_reach_err[t] = true_state.reach_err
        obs_qpos[t] = observed.qpos
        obs_qvel[t] = observed.qvel
        obs_act[t] = observed.act
        obs_tip_pos[t] = observed.tip_pos
        obs_target_pos[t] = observed.target_pos
        obs_reach_err[t] = observed.reach_err
        neural_command_buf[t] = neural_command
        excitation_command_buf[t] = excitation_command
        excitation_buf[t] = excitation
        api_action_buf[t] = api_action
        last_ctrl_buf[t] = last_ctrl
        reward_buf[t] = float(reward)
        terminated_buf[t] = bool(terminated)
        truncated_buf[t] = bool(truncated)
        x_est_buf[t] = x_est_flat.astype(_DT_F32, copy=False)

        prev_u = excitation.astype(_DT_F32, copy=False)
        n_steps = t + 1
        if terminated or truncated:
            break

    sl = slice(0, n_steps)
    sdn_sigma = float(sdn.sigma) if sdn is not None else 0.0
    obs_noise_sigma = dict(obs_noise.sigma) if obs_noise is not None else {}
    obs_delay_steps = int(obs_delay.delay_steps) if obs_delay is not None else 0

    log = EpisodeLog(
        episode_id=spec.episode_id,
        target_id=spec.target_id,
        target_split=spec.target_split,
        target_seed=spec.target_seed,
        target_pos_set=target_pos_arr,
        controller_name=spec.controller_name,
        controller_seed=spec.controller_seed,
        sdn_sigma=sdn_sigma,
        sdn_seed=spec.sdn_seed,
        obs_noise_sigma=obs_noise_sigma,
        obs_noise_seed=spec.obs_noise_seed,
        obs_delay_steps=obs_delay_steps,
        obs_compose=obs_compose,
        max_steps=T,
        n_steps=n_steps,
        created_at=datetime.now(timezone.utc).isoformat(),
        config_hash=spec.config_hash,
        step=step_buf[sl].copy(),
        time=time_buf[sl].copy(),
        true_qpos=true_qpos[sl].copy(),
        true_qvel=true_qvel[sl].copy(),
        true_act=true_act[sl].copy(),
        true_tip_pos=true_tip_pos[sl].copy(),
        true_target_pos=true_target_pos[sl].copy(),
        true_reach_err=true_reach_err[sl].copy(),
        obs_qpos=obs_qpos[sl].copy(),
        obs_qvel=obs_qvel[sl].copy(),
        obs_act=obs_act[sl].copy(),
        obs_tip_pos=obs_tip_pos[sl].copy(),
        obs_target_pos=obs_target_pos[sl].copy(),
        obs_reach_err=obs_reach_err[sl].copy(),
        neural_command=neural_command_buf[sl].copy(),
        excitation_command=excitation_command_buf[sl].copy(),
        excitation=excitation_buf[sl].copy(),
        api_action=api_action_buf[sl].copy(),
        last_ctrl=last_ctrl_buf[sl].copy(),
        reward=reward_buf[sl].copy(),
        terminated=terminated_buf[sl].copy(),
        truncated=truncated_buf[sl].copy(),
        meta=dict(spec.meta),
    )
    return ClosedLoopEpisodeResult(log=log, x_est=x_est_buf[sl].copy())
