"""Rollout pipeline: ``run_episode`` composes the layered components.

Pure function — given an env (assumed to come from ``factory.make_env``), a
controller, a target position, and the env-side adapters/wrappers, walk
the env for at most ``max_steps`` control steps and return an
``EpisodeLog`` with every layer's value recorded per step.

Layer composition (matches the boundaries set in earlier steps):

```text
true MyoArmState (oracle)
  ↓ obs pipeline (noisy ↔ delayed, order = obs_compose)
observed MyoArmState (controller-facing)
  ↓ controller.act
neural_command
  ↓ identity (placeholder for future neural-command -> excitation translation)
excitation_command
  ↓ SDN.apply  (signal-dependent motor noise; if None, identity + clip range)
excitation
  ↓ ActionAdapter.excitation_to_api_action
api_action
  ↓ env.step
mj_data update + reward + terminated + truncated
  ↓ extract_ctrl(env)  (post-step muscle ctrl that MyoSuite actually applied)
last_ctrl
```

Target injection at episode start uses the same ``mj_model.site_pos`` +
``mujoco.mj_forward`` strategy validated in Step 1 (``targets.py``). The
env's ``np_random`` is not relied on for target reproducibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import mujoco
import numpy as np

from myoarm_fse.controllers.base import Controller
from myoarm_fse.data.schema import EpisodeLog
from myoarm_fse.envs.actions import ActionAdapter
from myoarm_fse.envs.extractors import extract_ctrl, extract_state
from myoarm_fse.envs.noise import SignalDependentMotorNoise
from myoarm_fse.envs.state import MyoArmState, StateSpec
from myoarm_fse.envs.wrappers import (
    DelayedObservationWrapper,
    NoisyObservationWrapper,
)

_DT_F32 = np.dtype(np.float32)
_DT_INT = np.dtype(np.int64)
_DT_BOOL = np.dtype(np.bool_)
_CART_DIM: int = 3
_VALID_OBS_COMPOSE: tuple[str, ...] = ("noisy_then_delayed", "delayed_then_noisy")


@dataclass(frozen=True)
class EpisodeSpec:
    """Per-episode metadata that the rollout pipeline does not derive itself."""

    episode_id: int
    target_id: str
    target_split: str
    target_seed: int
    controller_name: str
    controller_seed: int
    sdn_seed: int = 0
    obs_noise_seed: int = 0
    config_hash: str = ""
    meta: dict[str, Any] = field(default_factory=dict)


def _apply_obs(
    true_state: MyoArmState,
    obs_noise: NoisyObservationWrapper | None,
    obs_delay: DelayedObservationWrapper | None,
    obs_compose: str,
) -> MyoArmState:
    """Apply the observation pipeline in the configured order."""
    if obs_compose == "noisy_then_delayed":
        x = obs_noise.observe(true_state) if obs_noise is not None else true_state
        return obs_delay.observe(x) if obs_delay is not None else x
    if obs_compose == "delayed_then_noisy":
        x = obs_delay.observe(true_state) if obs_delay is not None else true_state
        return obs_noise.observe(x) if obs_noise is not None else x
    raise ValueError(
        f"obs_compose must be one of {_VALID_OBS_COMPOSE}, got {obs_compose!r}"
    )


def _inject_target(env: Any, target_pos: np.ndarray) -> None:
    """Write ``target_pos`` to the env's target site via mj_model.site_pos."""
    uw = env.unwrapped
    if len(uw.target_sids) != 1:
        raise ValueError(
            f"expected exactly one target site, got target_sids={uw.target_sids!r}"
        )
    target_sid = int(uw.target_sids[0])
    uw.mj_model.site_pos[target_sid] = np.asarray(target_pos, dtype=np.float64)
    mujoco.mj_forward(uw.mj_model, uw.mj_data)


def run_episode(
    env: Any,
    controller: Controller,
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
) -> EpisodeLog:
    """Run one episode and return an ``EpisodeLog``.

    The env must come from ``factory.make_env``. Target reproducibility is
    handled by writing ``target_pos`` directly to ``mj_model.site_pos`` and
    calling ``mujoco.mj_forward``; the env's ``np_random`` is irrelevant.
    """
    if obs_compose not in _VALID_OBS_COMPOSE:
        raise ValueError(
            f"obs_compose must be one of {_VALID_OBS_COMPOSE}, got {obs_compose!r}"
        )
    if max_steps < 0:
        raise ValueError(f"max_steps must be >= 0, got {max_steps}")

    target_pos_arr = np.asarray(target_pos, dtype=_DT_F32).copy()
    if target_pos_arr.shape != (_CART_DIM,):
        raise ValueError(
            f"target_pos must have shape (3,), got {target_pos_arr.shape}"
        )

    if spec is None:
        spec = EpisodeSpec(
            episode_id=0,
            target_id="",
            target_split="",
            target_seed=0,
            controller_name=type(controller).__name__,
            controller_seed=0,
        )

    # --- env reset + target injection ---
    env.reset()
    _inject_target(env, target_pos_arr)
    initial_true_state = extract_state(env)

    # --- wrapper reset (delay buffer) ---
    if obs_delay is not None:
        obs_delay.reset(initial_true_state)

    # Sanity: state_spec must agree with the env's actual extraction.
    es = initial_true_state.spec()
    if (es.qpos_dim, es.qvel_dim, es.act_dim) != (
        state_spec.qpos_dim,
        state_spec.qvel_dim,
        state_spec.act_dim,
    ):
        raise ValueError(
            f"state_spec mismatch: env produces (qpos={es.qpos_dim}, "
            f"qvel={es.qvel_dim}, act={es.act_dim}) but state_spec has "
            f"({state_spec.qpos_dim}, {state_spec.qvel_dim}, {state_spec.act_dim})"
        )

    qpos_dim = state_spec.qpos_dim
    qvel_dim = state_spec.qvel_dim
    act_dim = state_spec.act_dim
    action_dim = action_adapter.action_dim
    dt = float(env.unwrapped.dt)
    T = max_steps

    # --- preallocate step buffers (sliced at the end to actual length) ---
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

    n_steps = 0

    for t in range(T):
        true_state = extract_state(env)
        observed = _apply_obs(true_state, obs_noise, obs_delay, obs_compose)

        neural_command = controller.act(observed)
        # neural_command -> excitation_command identity (current minimum design).
        excitation_command = np.asarray(neural_command, dtype=_DT_F32)
        if excitation_command.shape != (action_dim,):
            raise ValueError(
                f"controller returned shape {excitation_command.shape}, "
                f"expected ({action_dim},)"
            )
        if not np.isfinite(excitation_command).all():
            raise ValueError("controller produced non-finite excitation_command")
        if (excitation_command < 0.0).any() or (excitation_command > 1.0).any():
            raise ValueError(
                "controller produced excitation_command outside [0, 1]"
            )

        if sdn is not None:
            excitation = sdn.apply(excitation_command)
        else:
            excitation = action_adapter.clip_excitation(excitation_command)

        api_action = action_adapter.excitation_to_api_action(excitation)
        _, reward, terminated, truncated, _info = env.step(api_action)
        last_ctrl = extract_ctrl(env)

        # Record into pre-allocated buffers.
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

        n_steps = t + 1
        if terminated or truncated:
            break

    # Slice to actual length.
    sl = slice(0, n_steps)
    sdn_sigma = float(sdn.sigma) if sdn is not None else 0.0
    obs_noise_sigma = dict(obs_noise.sigma) if obs_noise is not None else {}
    obs_delay_steps = int(obs_delay.delay_steps) if obs_delay is not None else 0

    return EpisodeLog(
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
