"""Tests for myoarm_fse.metrics.reaching."""

from __future__ import annotations

import numpy as np
import pytest

from myoarm_fse.data import EpisodeLog
from myoarm_fse.metrics import (
    effort_norm,
    final_tip_error,
    minimum_tip_error,
    success,
)


# --- helpers ---


def _make_log(
    *,
    n_steps: int,
    reach_err_per_step: np.ndarray | None = None,
    excitation_per_step: np.ndarray | None = None,
    qpos_dim: int = 2,
    qvel_dim: int = 3,
    act_dim: int = 4,
    action_dim: int = 4,
    max_steps: int | None = None,
) -> EpisodeLog:
    T = n_steps
    f32 = np.float32
    if max_steps is None:
        max_steps = max(T, 1)

    def cart() -> np.ndarray:
        return np.zeros((T, 3), dtype=f32)

    if reach_err_per_step is None:
        true_reach_err = cart()
    else:
        true_reach_err = np.asarray(reach_err_per_step, dtype=f32)
        assert true_reach_err.shape == (T, 3), "reach_err_per_step must be (T, 3)"

    if excitation_per_step is None:
        excitation = np.zeros((T, action_dim), dtype=f32)
    else:
        excitation = np.asarray(excitation_per_step, dtype=f32)
        assert excitation.shape == (T, action_dim)

    return EpisodeLog(
        episode_id=0,
        target_id="train:0",
        target_split="train",
        target_seed=0,
        target_pos_set=np.zeros(3, dtype=f32),
        controller_name="test",
        controller_seed=0,
        sdn_sigma=0.0,
        sdn_seed=0,
        obs_noise_sigma={},
        obs_noise_seed=0,
        obs_delay_steps=0,
        obs_compose="noisy_then_delayed",
        max_steps=max_steps,
        n_steps=T,
        created_at="2026-05-10T00:00:00Z",
        config_hash="test",
        step=np.arange(T, dtype=np.int64),
        time=np.arange(T, dtype=f32) * 0.02,
        true_qpos=np.zeros((T, qpos_dim), dtype=f32),
        true_qvel=np.zeros((T, qvel_dim), dtype=f32),
        true_act=np.zeros((T, act_dim), dtype=f32),
        true_tip_pos=cart(),
        true_target_pos=cart(),
        true_reach_err=true_reach_err,
        obs_qpos=np.zeros((T, qpos_dim), dtype=f32),
        obs_qvel=np.zeros((T, qvel_dim), dtype=f32),
        obs_act=np.zeros((T, act_dim), dtype=f32),
        obs_tip_pos=cart(),
        obs_target_pos=cart(),
        obs_reach_err=cart(),
        neural_command=np.zeros((T, action_dim), dtype=f32),
        excitation_command=np.zeros((T, action_dim), dtype=f32),
        excitation=excitation,
        api_action=np.zeros((T, action_dim), dtype=f32),
        last_ctrl=np.zeros((T, action_dim), dtype=f32),
        reward=np.zeros(T, dtype=f32),
        terminated=np.zeros(T, dtype=np.bool_),
        truncated=np.zeros(T, dtype=np.bool_),
    )


# --- minimum_tip_error ---


class TestMinimumTipError:
    def test_zero_distance(self) -> None:
        log = _make_log(n_steps=5)  # all reach_err = 0
        assert minimum_tip_error(log) == 0.0

    def test_known_minimum(self) -> None:
        # First step distance 1.0, then 0.3, then 0.7 — min is 0.3.
        rerr = np.array(
            [[1.0, 0.0, 0.0], [0.3, 0.0, 0.0], [0.7, 0.0, 0.0]],
            dtype=np.float32,
        )
        log = _make_log(n_steps=3, reach_err_per_step=rerr)
        assert minimum_tip_error(log) == pytest.approx(0.3)

    def test_3d_norm(self) -> None:
        # ||[3, 4, 0]|| = 5
        rerr = np.array([[3.0, 4.0, 0.0]], dtype=np.float32)
        log = _make_log(n_steps=1, reach_err_per_step=rerr)
        assert minimum_tip_error(log) == pytest.approx(5.0)

    def test_empty_returns_inf(self) -> None:
        log = _make_log(n_steps=0, max_steps=10)
        assert minimum_tip_error(log) == float("inf")


# --- final_tip_error ---


class TestFinalTipError:
    def test_known_final(self) -> None:
        rerr = np.array(
            [[1.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.2, 0.0, 0.0]],
            dtype=np.float32,
        )
        log = _make_log(n_steps=3, reach_err_per_step=rerr)
        assert final_tip_error(log) == pytest.approx(0.2)

    def test_empty_returns_inf(self) -> None:
        log = _make_log(n_steps=0, max_steps=10)
        assert final_tip_error(log) == float("inf")

    def test_uses_last_only_not_minimum(self) -> None:
        rerr = np.array(
            [[0.01, 0.0, 0.0], [0.99, 0.0, 0.0]],
            dtype=np.float32,
        )
        log = _make_log(n_steps=2, reach_err_per_step=rerr)
        assert final_tip_error(log) == pytest.approx(0.99)


# --- success ---


class TestSuccess:
    def test_all_within_threshold_long_episode(self) -> None:
        rerr = np.zeros((20, 3), dtype=np.float32)
        log = _make_log(n_steps=20, reach_err_per_step=rerr)
        assert success(log, threshold=0.05, duration=10) is True

    def test_no_window_within_threshold(self) -> None:
        # All reach_err = 1.0, distance 1.0 > threshold.
        rerr = np.ones((20, 3), dtype=np.float32)
        log = _make_log(n_steps=20, reach_err_per_step=rerr)
        assert success(log, threshold=0.05, duration=10) is False

    def test_sustained_window_at_end(self) -> None:
        # First 10 outside threshold, last 10 inside.
        rerr = np.zeros((20, 3), dtype=np.float32)
        rerr[:10, 0] = 1.0
        log = _make_log(n_steps=20, reach_err_per_step=rerr)
        assert success(log, threshold=0.05, duration=10) is True

    def test_sustained_window_at_start(self) -> None:
        rerr = np.zeros((20, 3), dtype=np.float32)
        rerr[10:, 0] = 1.0
        log = _make_log(n_steps=20, reach_err_per_step=rerr)
        assert success(log, threshold=0.05, duration=10) is True

    def test_window_too_short_due_to_brief_excursion(self) -> None:
        # 9 within, 1 outside, 5 within → longest contiguous within-window is 9.
        rerr = np.zeros((15, 3), dtype=np.float32)
        rerr[9, 0] = 1.0
        log = _make_log(n_steps=15, reach_err_per_step=rerr)
        assert success(log, threshold=0.05, duration=10) is False

    def test_window_meets_exactly_duration(self) -> None:
        rerr = np.zeros((10, 3), dtype=np.float32)
        log = _make_log(n_steps=10, reach_err_per_step=rerr)
        assert success(log, threshold=0.05, duration=10) is True

    def test_episode_shorter_than_duration_returns_false(self) -> None:
        rerr = np.zeros((5, 3), dtype=np.float32)
        log = _make_log(n_steps=5, reach_err_per_step=rerr)
        assert success(log, threshold=0.05, duration=10) is False

    def test_threshold_boundary_strict_inequality(self) -> None:
        # err = exactly threshold should NOT count (strict <).
        rerr = np.full((10, 3), 0.05 / np.sqrt(3), dtype=np.float32)
        # ||rerr_t|| ≈ 0.05 (boundary)
        log = _make_log(n_steps=10, reach_err_per_step=rerr)
        # Strict <: should be False at boundary.
        assert success(log, threshold=0.05, duration=10) is False

    def test_default_arguments(self) -> None:
        rerr = np.zeros((20, 3), dtype=np.float32)
        log = _make_log(n_steps=20, reach_err_per_step=rerr)
        assert success(log) is True  # uses defaults

    def test_invalid_threshold_negative(self) -> None:
        log = _make_log(n_steps=10)
        with pytest.raises(ValueError):
            success(log, threshold=-0.01)

    def test_invalid_threshold_bool(self) -> None:
        log = _make_log(n_steps=10)
        with pytest.raises(ValueError):
            success(log, threshold=True)  # type: ignore[arg-type]

    def test_invalid_duration_zero(self) -> None:
        log = _make_log(n_steps=10)
        with pytest.raises(ValueError):
            success(log, duration=0)

    def test_invalid_duration_bool(self) -> None:
        log = _make_log(n_steps=10)
        with pytest.raises(ValueError):
            success(log, duration=True)  # type: ignore[arg-type]


# --- effort_norm ---


class TestEffortNorm:
    def test_zero_excitation(self) -> None:
        log = _make_log(n_steps=5)  # excitation = 0
        assert effort_norm(log) == 0.0

    def test_constant_excitation(self) -> None:
        # excitation = 0.5 across all 4 muscles, 5 steps
        # ||u_t||² = 4 * 0.25 = 1.0 for every t
        # mean over T = 1.0
        exc = np.full((5, 4), 0.5, dtype=np.float32)
        log = _make_log(n_steps=5, excitation_per_step=exc, action_dim=4)
        assert effort_norm(log) == pytest.approx(1.0)

    def test_per_step_squared_then_mean(self) -> None:
        # Step 0: u = [1, 0, 0, 0] → ||u||² = 1
        # Step 1: u = [0, 1, 0, 0] → ||u||² = 1
        # Step 2: u = [0.5, 0.5, 0, 0] → ||u||² = 0.5
        # mean = (1 + 1 + 0.5) / 3 = 0.833...
        exc = np.array(
            [[1.0, 0.0, 0.0, 0.0],
             [0.0, 1.0, 0.0, 0.0],
             [0.5, 0.5, 0.0, 0.0]],
            dtype=np.float32,
        )
        log = _make_log(n_steps=3, excitation_per_step=exc, action_dim=4)
        assert effort_norm(log) == pytest.approx(2.5 / 3)

    def test_empty_episode_returns_zero(self) -> None:
        log = _make_log(n_steps=0, max_steps=10)
        assert effort_norm(log) == 0.0
