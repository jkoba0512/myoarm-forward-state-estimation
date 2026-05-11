"""Tests for closed-loop metrics (Phase 2)."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from myoarm_fse.data.schema import EpisodeLog
from myoarm_fse.envs.state import StateSpec
from myoarm_fse.metrics import (
    closed_loop_episode_summary,
    max_tip_error,
    overshoot,
)


def _spec() -> StateSpec:
    return StateSpec(qpos_dim=2, qvel_dim=2, act_dim=3)


def _make_log(reach_err: np.ndarray, excitation: np.ndarray) -> EpisodeLog:
    """Build a minimal EpisodeLog with synthetic reach_err and excitation."""
    n = reach_err.shape[0]
    spec = _spec()
    action_dim = excitation.shape[1] if n > 0 else 3
    zeros = lambda dims: np.zeros((n, dims), dtype=np.float32)  # noqa: E731
    return EpisodeLog(
        episode_id=0, target_id="", target_split="", target_seed=0,
        target_pos_set=np.zeros(3, dtype=np.float32),
        controller_name="test", controller_seed=0,
        sdn_sigma=0.0, sdn_seed=0,
        obs_noise_sigma={}, obs_noise_seed=0,
        obs_delay_steps=0, obs_compose="noisy_then_delayed",
        max_steps=n, n_steps=n,
        created_at=datetime.now(timezone.utc).isoformat(),
        config_hash="",
        step=np.arange(n, dtype=np.int64),
        time=np.arange(n, dtype=np.float32) * 0.02,
        true_qpos=zeros(spec.qpos_dim),
        true_qvel=zeros(spec.qvel_dim),
        true_act=zeros(spec.act_dim),
        true_tip_pos=zeros(3),
        true_target_pos=zeros(3),
        true_reach_err=reach_err.astype(np.float32),
        obs_qpos=zeros(spec.qpos_dim),
        obs_qvel=zeros(spec.qvel_dim),
        obs_act=zeros(spec.act_dim),
        obs_tip_pos=zeros(3),
        obs_target_pos=zeros(3),
        obs_reach_err=zeros(3),
        neural_command=zeros(action_dim),
        excitation_command=zeros(action_dim),
        excitation=excitation.astype(np.float32),
        api_action=zeros(action_dim),
        last_ctrl=zeros(action_dim),
        reward=np.zeros(n, dtype=np.float32),
        terminated=np.zeros(n, dtype=np.bool_),
        truncated=np.zeros(n, dtype=np.bool_),
        meta={},
    )


# --- max_tip_error / overshoot ---


class TestMaxTipErrorAndOvershoot:
    def test_max_tip_error_basic(self) -> None:
        # reach_err norms: 0.1, 0.05, 0.2 -> max 0.2
        reach = np.array([[0.1, 0, 0], [0.05, 0, 0], [0.2, 0, 0]], dtype=np.float32)
        log = _make_log(reach, np.zeros((3, 3)))
        assert max_tip_error(log) == pytest.approx(0.2, abs=1e-6)

    def test_max_tip_error_empty(self) -> None:
        log = _make_log(np.zeros((0, 3)), np.zeros((0, 3)))
        assert max_tip_error(log) == 0.0

    def test_overshoot_basic(self) -> None:
        # norms: 0.2, 0.05, 0.1 -> max 0.2, final 0.1 -> overshoot 0.1
        reach = np.array([[0.2, 0, 0], [0.05, 0, 0], [0.1, 0, 0]], dtype=np.float32)
        log = _make_log(reach, np.zeros((3, 3)))
        assert overshoot(log) == pytest.approx(0.1, abs=1e-6)

    def test_overshoot_monotone_decrease_is_zero(self) -> None:
        reach = np.array([[0.3, 0, 0], [0.2, 0, 0], [0.1, 0, 0]], dtype=np.float32)
        log = _make_log(reach, np.zeros((3, 3)))
        # max is the first sample, final is the last -> overshoot = max - final = 0.2.
        assert overshoot(log) == pytest.approx(0.2, abs=1e-6)

    def test_overshoot_empty(self) -> None:
        log = _make_log(np.zeros((0, 3)), np.zeros((0, 3)))
        assert overshoot(log) == 0.0


# --- closed_loop_episode_summary ---


class TestClosedLoopEpisodeSummary:
    def _basic(self) -> tuple[EpisodeLog, np.ndarray, np.ndarray]:
        reach = np.array(
            [[0.20, 0, 0], [0.10, 0, 0], [0.04, 0, 0], [0.02, 0, 0]],
            dtype=np.float32,
        )
        exc = np.full((4, 3), 0.5, dtype=np.float32)
        log = _make_log(reach, exc)
        spec = _spec()
        # Make a trivially-perfect estimator: x_est == x_true. Layout
        # gives tip_pos at the offset after qpos+qvel+act = 2+2+3=7.
        T = log.n_steps
        x_true = np.zeros((T, spec.dim), dtype=np.float32)
        x_est = x_true.copy()
        return log, x_est, x_true

    def test_keys_and_basic_values(self) -> None:
        log, x_est, x_true = self._basic()
        spec = _spec()
        out = closed_loop_episode_summary(
            log, x_est=x_est, x_true=x_true, state_spec=spec,
        )
        # reaching
        assert out["n_steps"] == 4
        assert out["final_tip_error"] == pytest.approx(0.02, abs=1e-6)
        assert out["min_tip_error"] == pytest.approx(0.02, abs=1e-6)
        assert out["max_tip_error"] == pytest.approx(0.20, abs=1e-6)
        assert out["overshoot"] == pytest.approx(0.18, abs=1e-6)
        # effort_norm = mean over steps of ||excitation||^2 = 3 * 0.25 = 0.75
        assert out["effort_norm"] == pytest.approx(0.75, abs=1e-6)
        # success at 0.05 m needs 10 contiguous steps under threshold,
        # episode length is 4 -> always False.
        assert out["success_005"] is False
        assert out["success_010"] is False
        assert out["success_015"] is False
        # estimation: x_est == x_true exactly -> zero error.
        assert out["tip_estimation_error_mean"] == 0.0
        assert out["tip_estimation_error_final"] == 0.0
        assert out["state_mse_mean"] == 0.0

    def test_success_with_long_window(self) -> None:
        reach = np.tile(np.array([[0.01, 0, 0]], dtype=np.float32), (20, 1))
        log = _make_log(reach, np.zeros((20, 3)))
        spec = _spec()
        x_true = np.zeros((20, spec.dim), dtype=np.float32)
        out = closed_loop_episode_summary(
            log, x_est=x_true.copy(), x_true=x_true, state_spec=spec,
        )
        assert out["success_005"] is True
        assert out["success_010"] is True

    def test_estimation_error_nonzero(self) -> None:
        log, _, x_true = self._basic()
        spec = _spec()
        # Add a 0.01 m bias to only the first tip-pos coordinate so the
        # resulting per-step Euclidean error is exactly 0.01.
        x_est = x_true.copy()
        tip_slice = spec.layout()["tip_pos"]
        x_est[:, tip_slice.start] += 0.01
        out = closed_loop_episode_summary(
            log, x_est=x_est, x_true=x_true, state_spec=spec,
        )
        # tip_dist = ||[0.01, 0, 0]|| = 0.01 per step.
        assert out["tip_estimation_error_mean"] == pytest.approx(0.01, abs=1e-6)
        assert out["tip_estimation_error_final"] == pytest.approx(0.01, abs=1e-6)

    def test_skip_cold_start_trims(self) -> None:
        log, _, x_true = self._basic()
        spec = _spec()
        x_est = x_true.copy()
        tip_slice = spec.layout()["tip_pos"]
        # Inject bias only on first 2 steps; expect mean=0 after trimming.
        x_est[:2, tip_slice.start] += 0.05
        out_full = closed_loop_episode_summary(
            log, x_est=x_est, x_true=x_true, state_spec=spec,
            skip_cold_start_steps=0,
        )
        out_trim = closed_loop_episode_summary(
            log, x_est=x_est, x_true=x_true, state_spec=spec,
            skip_cold_start_steps=2,
        )
        assert out_full["tip_estimation_error_mean"] > 0.0
        assert out_trim["tip_estimation_error_mean"] == pytest.approx(0.0, abs=1e-6)

    def test_traj_shape_mismatch_raises(self) -> None:
        log, _, _ = self._basic()
        spec = _spec()
        # Both x_est and x_true wrong state_dim -> trajectory state dim check fires.
        with pytest.raises(ValueError, match="trajectory state dim"):
            closed_loop_episode_summary(
                log, x_est=np.zeros((log.n_steps, spec.dim + 1)),
                x_true=np.zeros((log.n_steps, spec.dim + 1)),
                state_spec=spec,
            )

    def test_traj_length_mismatch_raises(self) -> None:
        log, _, x_true = self._basic()
        spec = _spec()
        with pytest.raises(ValueError, match="x_est shape"):
            closed_loop_episode_summary(
                log, x_est=np.zeros((log.n_steps + 1, spec.dim)),
                x_true=x_true, state_spec=spec,
            )

    def test_non_log_raises(self) -> None:
        spec = _spec()
        with pytest.raises(ValueError, match="must be an EpisodeLog"):
            closed_loop_episode_summary(
                "not a log", x_est=np.zeros((1, spec.dim)),  # type: ignore[arg-type]
                x_true=np.zeros((1, spec.dim)), state_spec=spec,
            )
