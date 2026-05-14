"""Reliability-adaptive predictive state observer.

A neuroscience-motivated state observer that adapts its sensory
prediction-error correction gain from agent-available signals only,
without offline oracle access. The observer maintains a per-channel
(per-field) EMA of innovation variance, derives a reliability estimate
from it, and maps reliability to a correction gain via a logistic
function.

For each sensory channel ``f`` (qpos, qvel, act, tip_pos, reach_err)
the update at time ``t`` is::

    e_t[f]            = y_t[f] - x_pred_t[f]                       # innovation
    var_f[t]          = (1 - alpha) * var_f[t-1] + alpha * mean(e_t[f] ** 2)
    reliability_f[t]  = 1 / (epsilon + var_f[t])
    K_f[t]            = sigmoid(beta0_f + beta1_f * log(reliability_f[t]))

Then the per-channel correction gain ``K_f[t]`` is broadcast to the
83-dim gain vector ``gain_vec`` (one entry per field component) and
applied to the same innovation-style update as
``FixedGainKalmanEstimator``::

    xhat_t            = x_pred_t + gain_vec * (y_t - x_pred_t)

The ``target_pos`` field is intentionally excluded from reliability
adaptation: it is episode-constant in this benchmark and its
innovation is dominated by injection noise rather than sensory
variability. We hold its gain at a configurable constant
``target_pos_gain`` (default 1.0; the target is observed directly).

No information about the true state, the oracle gain, or any offline
K-sweep label is used at any point inside the observer. Within-trial
adaptation is purely driven by sensory prediction-error statistics
that the agent could in principle compute online.

This class shares the fixed-lag buffer machinery with
``FixedGainKalmanEstimator``: it predicts at the present time, corrects
the delayed buffer entry, and rolls the corrected past forward through
the cached actions ``u[t-d..t-1]``. The only difference is that the
gain vector is recomputed on every step from the running innovation
statistics, instead of being fixed at construction.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from myoarm_fse.envs.state import MyoArmState, StateSpec
from myoarm_fse.models.mlp import ForwardMLP

_DT_F32 = np.dtype(np.float32)

# Fields that participate in reliability adaptation. ``target_pos`` is
# excluded (episode-constant in our benchmark); its gain is held at a
# configurable constant so the user can still feed the target through
# the observer if they want.
_ADAPTIVE_FIELDS: tuple[str, ...] = ("qpos", "qvel", "act", "tip_pos", "reach_err")


@dataclass(frozen=True)
class ReliabilityAdaptiveConfig:
    """Hyperparameters for the reliability-adaptive observer.

    Defaults are chosen so that at the first step (when ``var_f`` is
    initialised to ``var_init``) the correction gain sits near ``0.5``,
    and the gain shifts smoothly toward ``1`` as reliability increases
    (innovation variance decreases).
    """

    alpha: float = 0.05
    """EMA decay for innovation variance. Effective window ~1/alpha."""

    epsilon: float = 1e-6
    """Small constant added to variance before inversion (numerical guard)."""

    var_init: float = 1.0
    """Initial EMA variance, applied to every adaptive field on reset."""

    beta0: dict[str, float] = field(
        default_factory=lambda: {f: 0.0 for f in _ADAPTIVE_FIELDS}
    )
    """Per-field bias term in K = sigmoid(beta0 + beta1 * log(reliability))."""

    beta1: dict[str, float] = field(
        default_factory=lambda: {f: 0.5 for f in _ADAPTIVE_FIELDS}
    )
    """Per-field slope on log(reliability)."""

    target_pos_gain: float = 1.0
    """Constant gain applied to the ``target_pos`` field (not adapted)."""


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


class ReliabilityAdaptiveObserver:
    """Forward-model-based predictive state observer with per-channel
    reliability-driven correction gain.

    The interface (``reset(initial_state)`` then ``step(y_obs, u)``)
    mirrors :class:`FixedGainKalmanEstimator`. ``delay_steps`` handling
    is identical (length-(d+1) ring buffer + length-d action buffer for
    re-rolling the corrected past forward).
    """

    def __init__(
        self,
        forward_model: ForwardMLP,
        state_spec: StateSpec,
        *,
        delay_steps: int = 0,
        config: ReliabilityAdaptiveConfig | None = None,
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
        self._cfg = config or ReliabilityAdaptiveConfig()

        # Cache field slices and validate that all adaptive fields are present.
        self._layout = state_spec.layout()
        for fname in _ADAPTIVE_FIELDS:
            if fname not in self._layout:
                raise ValueError(
                    f"state_spec layout missing field {fname!r}; "
                    f"got {list(self._layout)}"
                )
        for fname in _ADAPTIVE_FIELDS:
            if fname not in self._cfg.beta0 or fname not in self._cfg.beta1:
                raise ValueError(
                    f"config beta0/beta1 missing entry for field {fname!r}"
                )

        # Buffers populated on reset.
        self._x_buffer: deque[np.ndarray] | None = None
        self._u_buffer: deque[np.ndarray] | None = None
        self._var: dict[str, float] | None = None

        # Per-step history (cleared on reset). Kept in pure Python lists
        # because the trajectory length is set by the caller; convert to
        # arrays at the end of the episode if needed.
        self._innovation_history: list[np.ndarray] = []
        self._var_history: list[dict[str, float]] = []
        self._reliability_history: list[dict[str, float]] = []
        self._k_history: list[dict[str, float]] = []

    # --- properties ---

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
    def config(self) -> ReliabilityAdaptiveConfig:
        return self._cfg

    @property
    def adaptive_fields(self) -> tuple[str, ...]:
        return _ADAPTIVE_FIELDS

    # --- history accessors ---

    def innovation_history(self) -> np.ndarray:
        """Return ``(T, state_dim)`` per-step innovations recorded so far."""
        if not self._innovation_history:
            return np.empty((0, self.state_dim), dtype=_DT_F32)
        return np.stack(self._innovation_history, axis=0)

    @property
    def k_history(self) -> list[float]:
        """Mean correction gain across adaptive fields per step (read-only).

        Returns a flat list of scalars (one per recorded step) to match
        the interface used by :class:`LearnedGainKalmanEstimator` and
        consumed by the closed-loop logger. For field-wise detail use
        :meth:`field_k_history`.
        """
        if not self._k_history:
            return []
        return [
            float(np.mean(list(row.values()))) for row in self._k_history
        ]

    def field_k_history(self) -> dict[str, np.ndarray]:
        """Return ``{field: (T,) array}`` of per-step correction gains."""
        return self._history_to_arrays(self._k_history)

    def field_reliability_history(self) -> dict[str, np.ndarray]:
        return self._history_to_arrays(self._reliability_history)

    def field_var_history(self) -> dict[str, np.ndarray]:
        return self._history_to_arrays(self._var_history)

    @staticmethod
    def _history_to_arrays(
        history: list[dict[str, float]],
    ) -> dict[str, np.ndarray]:
        if not history:
            return {f: np.empty((0,), dtype=_DT_F32) for f in _ADAPTIVE_FIELDS}
        out: dict[str, np.ndarray] = {}
        for f in _ADAPTIVE_FIELDS:
            out[f] = np.array(
                [row.get(f, np.nan) for row in history], dtype=_DT_F32
            )
        return out

    # --- main API ---

    def reset(self, initial_state: np.ndarray) -> None:
        arr = np.asarray(initial_state, dtype=_DT_F32)
        if arr.shape != (self.state_dim,):
            raise ValueError(
                f"initial_state must have shape ({self.state_dim},), "
                f"got {arr.shape}"
            )
        if not np.isfinite(arr).all():
            raise ValueError("initial_state contains non-finite values")

        self._x_buffer = deque(
            [arr.copy() for _ in range(self._delay_steps + 1)],
            maxlen=self._delay_steps + 1,
        )
        self._u_buffer = deque(maxlen=self._delay_steps)

        self._var = {f: float(self._cfg.var_init) for f in _ADAPTIVE_FIELDS}

        self._innovation_history.clear()
        self._var_history.clear()
        self._reliability_history.clear()
        self._k_history.clear()

    def step(self, y_obs: np.ndarray, u: np.ndarray) -> np.ndarray:
        if (
            self._x_buffer is None
            or self._u_buffer is None
            or self._var is None
        ):
            raise RuntimeError(
                "ReliabilityAdaptiveObserver.step called before reset; "
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

        # Snapshot u_buffer BEFORE we push u_arr — the rollout for the
        # delayed branch needs u[t-d..t-1] without u[t].
        past_actions = list(self._u_buffer)

        # Predict from the most recent estimate using the just-applied u.
        current = self._x_buffer[-1]
        x_pred_next = current + self._predict_delta(current, u_arr)

        if self._delay_steps == 0:
            # Innovation is computed against the present-time prediction.
            innovation = y_arr - x_pred_next
            gain_vec = self._update_gain_from_innovation(innovation)
            corrected = x_pred_next + gain_vec * innovation
            self._record_innovation(innovation)
            self._x_buffer[-1] = corrected.astype(_DT_F32, copy=False)
            return corrected.astype(_DT_F32, copy=True)

        # delay > 0
        if len(past_actions) < self._delay_steps:
            # Cold start: not enough past actions to roll forward yet.
            # We still observe an innovation against the past buffer
            # entry (which represents our estimate at time t-d), so the
            # reliability EMA can warm up; but the correction is not
            # applied to the present-time estimate.
            x_at_past = self._x_buffer[0]
            innovation = y_arr - x_at_past
            self._update_gain_from_innovation(innovation)
            self._record_innovation(innovation)
            self._x_buffer.append(x_pred_next.astype(_DT_F32, copy=False))
            self._u_buffer.append(u_arr.copy())
            return x_pred_next.astype(_DT_F32, copy=True)

        # delay > 0 and buffer warmed up:
        x_at_past = self._x_buffer[0]
        innovation = y_arr - x_at_past
        gain_vec = self._update_gain_from_innovation(innovation)
        corrected_past = x_at_past + gain_vec * innovation

        # Roll the corrected past forward through past_actions
        # (= u[t-d..t-1]) to recover the estimate at time t.
        x = corrected_past
        for past_u in past_actions:
            x = x + self._predict_delta(x, past_u)

        self._record_innovation(innovation)
        self._x_buffer.append(x.astype(_DT_F32, copy=False))
        self._u_buffer.append(u_arr.copy())
        return x.astype(_DT_F32, copy=True)

    # --- internals ---

    def _update_gain_from_innovation(
        self, innovation: np.ndarray
    ) -> np.ndarray:
        """Update the EMA variance per field and emit a 83-dim gain vector.

        Side effect: appends a row to var/reliability/k history.
        """
        assert self._var is not None  # noqa: S101 — invariant after reset
        alpha = self._cfg.alpha
        eps = self._cfg.epsilon

        var_row: dict[str, float] = {}
        rel_row: dict[str, float] = {}
        k_row: dict[str, float] = {}

        gain_vec = np.zeros(self.state_dim, dtype=_DT_F32)
        for fname in _ADAPTIVE_FIELDS:
            sl = self._layout[fname]
            e_f = innovation[sl]
            # Mean squared innovation across the field's dims.
            mse = float(np.mean(e_f * e_f))
            new_var = (1.0 - alpha) * self._var[fname] + alpha * mse
            self._var[fname] = new_var
            reliability = 1.0 / (eps + new_var)
            log_r = float(np.log(reliability))
            z = self._cfg.beta0[fname] + self._cfg.beta1[fname] * log_r
            k = float(_sigmoid(np.array([z], dtype=np.float64))[0])

            gain_vec[sl] = np.float32(k)

            var_row[fname] = new_var
            rel_row[fname] = reliability
            k_row[fname] = k

        # target_pos: constant gain (not adapted).
        tgt_sl = self._layout.get("target_pos")
        if tgt_sl is not None:
            gain_vec[tgt_sl] = np.float32(self._cfg.target_pos_gain)

        self._var_history.append(var_row)
        self._reliability_history.append(rel_row)
        self._k_history.append(k_row)
        return gain_vec

    def _record_innovation(self, innovation: np.ndarray) -> None:
        self._innovation_history.append(innovation.astype(_DT_F32, copy=True))

    def _predict_delta(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        x_t = torch.from_numpy(x).unsqueeze(0)
        u_t = torch.from_numpy(u).unsqueeze(0)
        with torch.no_grad():
            dx = self._forward_model(x_t, u_t).squeeze(0).numpy()
        return dx.astype(_DT_F32, copy=False)
