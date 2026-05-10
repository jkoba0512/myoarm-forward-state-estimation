"""Fixed-gain Kalman-like state estimator (Phase 3.1).

Implements the Project 1 §3.1 update::

    x_pred = f(x_est, u)                  # forward model = trained ForwardMLP
    y_obs  = h(x_true_{t-d}) + noise      # delayed/noisy observation
    x_est  = x_pred + K * (y_obs - h(x_pred_{t-d}))  # buffered version

with ``h = identity`` (Step 2 / Step 4 schema) and ``K`` either a scalar
or a per-field dict expanded internally to a ``(state_dim,)`` weight
vector. ``K = 0`` is prediction-only; ``K = 1`` is observation-only.

Delay handling is the **fixed-lag Kalman smoother** minimum form. At
each step we:

1. Snapshot the current estimate (representing ``x[t-1]``).
2. Push a fresh prediction ``x_pred[t]`` to the buffer of recent
   estimates (length ``delay_steps + 1``).
3. If ``delay_steps == 0``, run the standard one-step update against
   the just-pushed prediction.
4. Otherwise, the new ``y_obs`` corresponds to ``x_true[t - delay_steps]``.
   We compare it against the buffer's leftmost entry (which represents
   our prior estimate at that earlier time), correct that entry, and
   roll it forward through the cached actions ``u[t-d..t-1]`` to obtain
   the corrected estimate at the current time. During the cold-start
   period (when ``len(u_buffer) < delay_steps``) we simply propagate
   the prediction without correction.

Caller convention: ``reset(initial_state)`` seeds ``x_est`` for time
``t = 0``. Subsequent calls ``step(y_obs[t], u[t])`` advance the
estimator by one tick. ``u[t]`` is the action applied between
``t-1`` and ``t``; the returned value is the best estimate of state at
time ``t`` after the update.

Offline evaluation helpers:

- :func:`synth_observations` — replay Step 4 wrappers over a recorded
  ``EpisodeLog.true_*`` to synthesize ``y_obs`` for any
  ``(sigma, delay)`` combination without re-running MyoSuite.
- :func:`evaluate_estimator_on_log` — run an estimator over one log,
  returning per-step error arrays.
- :func:`aggregate_estimation_metrics` — collapse a batch of
  ``EstimationResult``s to a JSON-serializable per-field summary.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from myoarm_fse.data.schema import EpisodeLog
from myoarm_fse.envs.state import MyoArmState, StateSpec
from myoarm_fse.envs.wrappers import (
    DelayedObservationWrapper,
    NoisyObservationWrapper,
)
from myoarm_fse.models.mlp import ForwardMLP

_DT_F32 = np.dtype(np.float32)
_K_LO: float = 0.0
_K_HI: float = 1.0
_VALID_OBS_COMPOSE: tuple[str, ...] = ("noisy_then_delayed", "delayed_then_noisy")


# --- gain validation ---


def _check_gain_value(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.floating)):
        raise ValueError(
            f"{name} must be numeric, got {type(value).__name__}: {value!r}"
        )
    v = float(value)
    if not np.isfinite(v):
        raise ValueError(f"{name} must be finite, got {v}")
    if v < _K_LO or v > _K_HI:
        raise ValueError(
            f"{name} must be within [{_K_LO}, {_K_HI}], got {v}"
        )
    return v


def _build_gain_vector(
    gain: float | dict[str, float],
    state_spec: StateSpec,
) -> np.ndarray:
    """Expand ``gain`` to a ``(state_dim,)`` element-wise weight vector."""
    layout = state_spec.layout()
    state_dim = state_spec.dim
    if isinstance(gain, dict):
        unknown = set(gain.keys()) - set(layout.keys())
        if unknown:
            raise ValueError(
                f"unknown gain field names: {sorted(unknown)}; "
                f"valid: {sorted(layout.keys())}"
            )
        vec = np.zeros(state_dim, dtype=_DT_F32)
        for name, slc in layout.items():
            v = _check_gain_value(gain.get(name, 0.0), f"gain[{name!r}]")
            vec[slc] = v
        return vec
    v = _check_gain_value(gain, "gain")
    return np.full(state_dim, v, dtype=_DT_F32)


# --- estimator ---


class FixedGainKalmanEstimator:
    """Fixed-gain Kalman-like estimator with optional fixed observation delay."""

    def __init__(
        self,
        forward_model: ForwardMLP,
        gain: float | dict[str, float],
        state_spec: StateSpec,
        *,
        delay_steps: int = 0,
    ) -> None:
        if not isinstance(forward_model, ForwardMLP):
            raise ValueError(
                f"forward_model must be ForwardMLP, "
                f"got {type(forward_model).__name__}"
            )
        if not isinstance(state_spec, StateSpec):
            raise TypeError(
                f"state_spec must be StateSpec, got {type(state_spec).__name__}"
            )
        if isinstance(delay_steps, bool) or not isinstance(delay_steps, int):
            raise ValueError(
                f"delay_steps must be a non-negative int, got {delay_steps!r}"
            )
        if delay_steps < 0:
            raise ValueError(f"delay_steps must be >= 0, got {delay_steps}")
        if forward_model.state_dim != state_spec.dim:
            raise ValueError(
                f"forward_model.state_dim={forward_model.state_dim} does not "
                f"match state_spec.dim={state_spec.dim}"
            )

        self._forward_model = forward_model
        self._state_spec = state_spec
        self._delay_steps = int(delay_steps)
        self._gain_vec = _build_gain_vector(gain, state_spec)

        # Populated on reset; None signals "must reset before step".
        self._x_buffer: deque[np.ndarray] | None = None
        self._u_buffer: deque[np.ndarray] | None = None

    @property
    def state_dim(self) -> int:
        return int(self._state_spec.dim)

    @property
    def action_dim(self) -> int:
        return int(self._forward_model.action_dim)

    @property
    def delay_steps(self) -> int:
        return self._delay_steps

    @property
    def gain_vec(self) -> np.ndarray:
        """Internal element-wise gain vector (read-only copy)."""
        return self._gain_vec.copy()

    def reset(self, initial_state: np.ndarray) -> None:
        arr = np.asarray(initial_state, dtype=_DT_F32)
        if arr.shape != (self.state_dim,):
            raise ValueError(
                f"initial_state must have shape ({self.state_dim},), "
                f"got {arr.shape}"
            )
        if not np.isfinite(arr).all():
            raise ValueError("initial_state contains non-finite values")
        # Buffer always has length delay_steps + 1; rightmost slot is
        # the current estimate. delay=0 still uses a length-1 buffer
        # for uniformity.
        self._x_buffer = deque(
            [arr.copy() for _ in range(self._delay_steps + 1)],
            maxlen=self._delay_steps + 1,
        )
        # u_buffer holds the actions u[t-d..t-1] needed to roll a
        # corrected past estimate forward to the present. Empty after
        # reset (cold start).
        self._u_buffer = deque(maxlen=self._delay_steps)

    def step(self, y_obs: np.ndarray, u: np.ndarray) -> np.ndarray:
        if self._x_buffer is None or self._u_buffer is None:
            raise RuntimeError(
                "FixedGainKalmanEstimator.step called before reset; "
                "call reset(initial_state) at the start of each episode"
            )
        y_arr = np.asarray(y_obs, dtype=_DT_F32)
        if y_arr.shape != (self.state_dim,):
            raise ValueError(
                f"y_obs must have shape ({self.state_dim},), got {y_arr.shape}"
            )
        if not np.isfinite(y_arr).all():
            raise ValueError("y_obs contains non-finite values")
        u_arr = np.asarray(u, dtype=_DT_F32)
        if u_arr.shape != (self.action_dim,):
            raise ValueError(
                f"u must have shape ({self.action_dim},), got {u_arr.shape}"
            )

        # Snapshot u_buffer BEFORE we push u_arr — the rollout in the
        # delayed branch needs the actions u[t-d..t-1] without u[t].
        past_actions = list(self._u_buffer)

        # Predict from the most recent estimate using the just-applied u.
        current = self._x_buffer[-1]
        x_pred_next = current + self._predict_delta(current, u_arr)

        if self._delay_steps == 0:
            # Standard one-step update: y_obs[t] is a noisy view of
            # x_true[t], compared directly with the prediction.
            innovation = y_arr - x_pred_next
            corrected = x_pred_next + self._gain_vec * innovation
            self._x_buffer[-1] = corrected.astype(_DT_F32, copy=False)
            return corrected.astype(_DT_F32, copy=True)

        # delay > 0
        if len(past_actions) < self._delay_steps:
            # Cold start: not enough past actions to roll forward yet,
            # so propagate prediction without correction.
            self._x_buffer.append(x_pred_next.astype(_DT_F32, copy=False))
            self._u_buffer.append(u_arr.copy())
            return x_pred_next.astype(_DT_F32, copy=True)

        # x_buffer[0] is the prior estimate at time t - delay_steps
        # (because the buffer represented [t-d-1+1, ..., t-1] = [t-d, ..., t-1]
        # before the push that we are about to do). y_obs[t] gives us a
        # noisy view of x_true[t - delay_steps]; correct buffer[0] with
        # that.
        x_at_past = self._x_buffer[0]
        innovation = y_arr - x_at_past
        corrected_past = x_at_past + self._gain_vec * innovation

        # Roll the corrected past forward through past_actions
        # (= u[t-d..t-1]) to recover the estimate at time t.
        x = corrected_past
        for past_u in past_actions:
            x = x + self._predict_delta(x, past_u)

        # Push the corrected current estimate; deque maxlen drops the
        # oldest entry so the buffer represents [t-d+1, ..., t].
        self._x_buffer.append(x.astype(_DT_F32, copy=False))
        self._u_buffer.append(u_arr.copy())
        return x.astype(_DT_F32, copy=True)

    # --- internal ---

    def _predict_delta(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        x_t = torch.from_numpy(x).unsqueeze(0)
        u_t = torch.from_numpy(u).unsqueeze(0)
        with torch.no_grad():
            dx = self._forward_model(x_t, u_t).squeeze(0).numpy()
        return dx.astype(_DT_F32, copy=False)


# --- offline evaluation helpers ---


def _flatten_log_states(log: EpisodeLog) -> np.ndarray:
    """Return ``(T, state_dim)`` flat true states matching ``MyoArmState.flatten()`` order."""
    T = log.n_steps
    if T == 0:
        return np.empty((0, 0), dtype=_DT_F32)
    return np.concatenate(
        [
            log.true_qpos,
            log.true_qvel,
            log.true_act,
            log.true_tip_pos,
            log.true_target_pos,
            log.true_reach_err,
        ],
        axis=1,
        dtype=_DT_F32,
    )


def _state_at(log: EpisodeLog, t: int) -> MyoArmState:
    return MyoArmState.from_arrays(
        qpos=log.true_qpos[t],
        qvel=log.true_qvel[t],
        act=log.true_act[t],
        tip_pos=log.true_tip_pos[t],
        target_pos=log.true_target_pos[t],
        reach_err=log.true_reach_err[t],
    )


def synth_observations(
    log: EpisodeLog,
    *,
    state_spec: StateSpec,
    sigma: dict[str, float],
    delay_steps: int,
    seed: int,
    obs_compose: str = "noisy_then_delayed",
) -> np.ndarray:
    """Re-apply Step 4 wrappers to a log's true_state to synthesize y_obs.

    Returns a ``(T, state_dim)`` ndarray with the same composition order
    as the rollout pipeline (default ``noisy_then_delayed``).
    """
    if not isinstance(log, EpisodeLog):
        raise ValueError("log must be EpisodeLog")
    if obs_compose not in _VALID_OBS_COMPOSE:
        raise ValueError(
            f"obs_compose must be one of {_VALID_OBS_COMPOSE}, got {obs_compose!r}"
        )
    T = log.n_steps
    y_obs = np.empty((T, state_spec.dim), dtype=_DT_F32)
    if T == 0:
        return y_obs

    noise = NoisyObservationWrapper(spec=state_spec, sigma=sigma, rng=seed)
    delay = DelayedObservationWrapper(spec=state_spec, delay_steps=delay_steps)
    delay.reset(_state_at(log, 0))

    for t in range(T):
        true_state = _state_at(log, t)
        if obs_compose == "noisy_then_delayed":
            observed = delay.observe(noise.observe(true_state))
        else:
            observed = noise.observe(delay.observe(true_state))
        y_obs[t] = observed.flatten()
    return y_obs


@dataclass(frozen=True, eq=False)
class EstimationResult:
    """Per-episode output of :func:`evaluate_estimator_on_log`."""

    x_est: np.ndarray
    x_true: np.ndarray
    error_per_step: np.ndarray
    error_per_step_norm: np.ndarray
    n_steps: int
    delay_steps: int
    layout: dict[str, slice] = field(default_factory=dict)


def evaluate_estimator_on_log(
    estimator: FixedGainKalmanEstimator,
    log: EpisodeLog,
    *,
    state_spec: StateSpec,
    obs_noise_sigma: dict[str, float],
    obs_delay_steps: int,
    obs_noise_seed: int,
    obs_compose: str = "noisy_then_delayed",
) -> EstimationResult:
    """Run estimator over one episode log and report per-step error.

    The estimator's own ``delay_steps`` should match ``obs_delay_steps``
    or behavior is unspecified (we don't enforce equality so test code
    can simulate misaligned conditions).
    """
    if not isinstance(log, EpisodeLog):
        raise ValueError("log must be EpisodeLog")
    if log.n_steps == 0:
        empty = np.empty((0, state_spec.dim), dtype=_DT_F32)
        return EstimationResult(
            x_est=empty,
            x_true=empty,
            error_per_step=empty,
            error_per_step_norm=np.empty((0,), dtype=_DT_F32),
            n_steps=0,
            delay_steps=obs_delay_steps,
            layout=state_spec.layout(),
        )

    y_obs = synth_observations(
        log,
        state_spec=state_spec,
        sigma=obs_noise_sigma,
        delay_steps=obs_delay_steps,
        seed=obs_noise_seed,
        obs_compose=obs_compose,
    )
    x_true = _flatten_log_states(log)
    u = log.excitation.astype(_DT_F32, copy=False)
    initial_state = x_true[0]

    estimator.reset(initial_state)
    x_est = np.empty_like(x_true)
    for t in range(log.n_steps):
        x_est[t] = estimator.step(y_obs[t], u[t])

    err = (x_est - x_true).astype(_DT_F32, copy=False)
    err_norm = np.linalg.norm(err, axis=1).astype(_DT_F32, copy=False)
    return EstimationResult(
        x_est=x_est,
        x_true=x_true,
        error_per_step=err,
        error_per_step_norm=err_norm,
        n_steps=log.n_steps,
        delay_steps=obs_delay_steps,
        layout=state_spec.layout(),
    )


def aggregate_estimation_metrics(
    results: Iterable[EstimationResult],
    *,
    skip_cold_start: bool = True,
) -> dict[str, Any]:
    """Per-run summary of per-field MSE + tip estimation error."""
    result_list = list(results)
    if not result_list:
        return {"n": 0}

    layout = result_list[0].layout
    per_field_mse: dict[str, list[float]] = {name: [] for name in layout}
    tip_err_list: list[float] = []
    n_total = 0

    for r in result_list:
        start = r.delay_steps if skip_cold_start else 0
        if r.n_steps <= start:
            continue
        n_total += 1
        err = r.error_per_step[start:]
        for name, slc in layout.items():
            per_field_mse[name].append(float(np.mean(err[:, slc] ** 2)))
        tip_slice = layout["tip_pos"]
        tip_dist = np.linalg.norm(err[:, tip_slice], axis=1)
        tip_err_list.append(float(np.mean(tip_dist)))

    out: dict[str, Any] = {"n": n_total}
    for name, vals in per_field_mse.items():
        if vals:
            out[f"mse_{name}_mean"] = float(np.mean(vals))
            out[f"mse_{name}_std"] = float(np.std(vals))
        else:
            out[f"mse_{name}_mean"] = 0.0
            out[f"mse_{name}_std"] = 0.0
    if tip_err_list:
        out["tip_estimation_error_mean"] = float(np.mean(tip_err_list))
        out["tip_estimation_error_std"] = float(np.std(tip_err_list))
    else:
        out["tip_estimation_error_mean"] = 0.0
        out["tip_estimation_error_std"] = 0.0
    return out
