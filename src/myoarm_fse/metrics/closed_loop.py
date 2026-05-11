"""Closed-loop episode metrics (Phase 2).

Adds ``max_tip_error`` and ``overshoot`` on top of the open-loop reaching
metrics, plus a ``ClosedLoopEpisodeMetrics`` builder that combines tip
metrics from a logged episode with estimation-quality metrics from
matched ``(x_est, x_true)`` trajectories. Pure numpy/numpy-compatible
inputs — no torch, no env dependencies — so the function is unit-
testable without MyoSuite.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from myoarm_fse.data.schema import EpisodeLog
from myoarm_fse.envs.state import StateSpec
from myoarm_fse.metrics.reaching import (
    effort_norm,
    final_tip_error,
    minimum_tip_error,
    success,
)

_CART_DIM: int = 3


def max_tip_error(log: EpisodeLog) -> float:
    """Largest ``||tip - target||`` reached anywhere in the episode.

    Returns ``0.0`` for an empty episode (no steps).
    """
    if log.n_steps == 0:
        return 0.0
    return float(np.max(np.linalg.norm(log.true_reach_err, axis=1)))


def overshoot(log: EpisodeLog) -> float:
    """Peak-minus-final reach-error magnitude.

    A simple definition: ``max(||tip - target||) - ||final tip - target||``.
    Always non-negative. Returns ``0.0`` for an empty episode.

    Not a true overshoot in the control-theoretic sense (we don't detect
    crossing the target), but captures whether the controller approaches
    then drifts away.
    """
    if log.n_steps == 0:
        return 0.0
    norms = np.linalg.norm(log.true_reach_err, axis=1)
    return float(norms.max() - norms[-1])


def _validate_traj_pair(
    x_est: np.ndarray, x_true: np.ndarray, state_dim: int
) -> None:
    if x_est.ndim != 2 or x_true.ndim != 2:
        raise ValueError("x_est and x_true must be 2-D (T, state_dim)")
    if x_est.shape != x_true.shape:
        raise ValueError(
            f"x_est shape {x_est.shape} != x_true shape {x_true.shape}"
        )
    if x_est.shape[1] != state_dim:
        raise ValueError(
            f"trajectory state dim {x_est.shape[1]} != state_spec.dim {state_dim}"
        )


def closed_loop_episode_summary(
    log: EpisodeLog,
    *,
    x_est: np.ndarray,
    x_true: np.ndarray,
    state_spec: StateSpec,
    success_thresholds: tuple[float, ...] = (0.05, 0.10, 0.15),
    success_duration: int = 10,
    skip_cold_start_steps: int = 0,
) -> dict[str, Any]:
    """Build a flat dict of closed-loop metrics for one episode.

    Reaching metrics (``final_tip_error``, ``min_tip_error``,
    ``max_tip_error``, ``overshoot``, ``effort_norm`` and per-threshold
    success flags) read the episode log's ``true_reach_err``.
    Estimation metrics aggregate ``x_est`` against ``x_true`` (both
    ``(T, state_dim)``) using the layout from ``state_spec``.

    ``skip_cold_start_steps`` lets the caller drop the first N samples
    of the estimation trajectory (e.g., the cold-start window of a
    delayed estimator). Reaching metrics are computed on the full
    trajectory.
    """
    if not isinstance(log, EpisodeLog):
        raise ValueError("log must be an EpisodeLog")
    if not isinstance(state_spec, StateSpec):
        raise ValueError("state_spec must be a StateSpec")
    x_est = np.asarray(x_est, dtype=np.float32)
    x_true = np.asarray(x_true, dtype=np.float32)
    _validate_traj_pair(x_est, x_true, state_spec.dim)
    if skip_cold_start_steps < 0:
        raise ValueError(
            f"skip_cold_start_steps must be >= 0, got {skip_cold_start_steps}"
        )
    if skip_cold_start_steps > x_est.shape[0]:
        skip_cold_start_steps = x_est.shape[0]

    out: dict[str, Any] = {
        "n_steps": int(log.n_steps),
        "final_tip_error": float(final_tip_error(log)),
        "min_tip_error": float(minimum_tip_error(log)),
        "max_tip_error": float(max_tip_error(log)),
        "overshoot": float(overshoot(log)),
        "effort_norm": float(effort_norm(log)),
    }
    for thr in success_thresholds:
        # Keys like success_005 / success_010 / success_015.
        key = f"success_{int(round(thr * 100)):03d}"
        out[key] = bool(
            success(log, threshold=float(thr), duration=success_duration)
        )

    # Estimation metrics on the (possibly cold-start-trimmed) trajectory.
    if x_est.shape[0] == 0 or skip_cold_start_steps == x_est.shape[0]:
        out["tip_estimation_error_mean"] = 0.0
        out["tip_estimation_error_final"] = 0.0
        out["state_mse_mean"] = 0.0
    else:
        layout = state_spec.layout()
        tip_slice = layout["tip_pos"]
        err = x_est[skip_cold_start_steps:] - x_true[skip_cold_start_steps:]
        tip_dist = np.linalg.norm(err[:, tip_slice], axis=1)
        out["tip_estimation_error_mean"] = float(np.mean(tip_dist))
        out["tip_estimation_error_final"] = float(tip_dist[-1])
        out["state_mse_mean"] = float(np.mean(err ** 2))
    return out
