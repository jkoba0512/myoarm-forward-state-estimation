"""Tests for myoarm_fse.metrics.aggregate."""

from __future__ import annotations

import numpy as np
import pytest

from myoarm_fse.data import EpisodeLog
from myoarm_fse.metrics import aggregate_reaching


def _log(
    *,
    n_steps: int = 20,
    constant_reach_dist: float = 0.0,
    constant_excitation: float = 0.0,
    action_dim: int = 4,
) -> EpisodeLog:
    T = n_steps
    f32 = np.float32
    rerr = np.zeros((T, 3), dtype=f32)
    rerr[:, 0] = constant_reach_dist  # ||rerr|| = constant_reach_dist
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
        max_steps=max(T, 1),
        n_steps=T,
        created_at="2026-05-10T00:00:00Z",
        config_hash="test",
        step=np.arange(T, dtype=np.int64),
        time=np.arange(T, dtype=f32) * 0.02,
        true_qpos=np.zeros((T, 2), dtype=f32),
        true_qvel=np.zeros((T, 3), dtype=f32),
        true_act=np.zeros((T, action_dim), dtype=f32),
        true_tip_pos=np.zeros((T, 3), dtype=f32),
        true_target_pos=np.zeros((T, 3), dtype=f32),
        true_reach_err=rerr,
        obs_qpos=np.zeros((T, 2), dtype=f32),
        obs_qvel=np.zeros((T, 3), dtype=f32),
        obs_act=np.zeros((T, action_dim), dtype=f32),
        obs_tip_pos=np.zeros((T, 3), dtype=f32),
        obs_target_pos=np.zeros((T, 3), dtype=f32),
        obs_reach_err=np.zeros((T, 3), dtype=f32),
        neural_command=np.zeros((T, action_dim), dtype=f32),
        excitation_command=np.zeros((T, action_dim), dtype=f32),
        excitation=np.full((T, action_dim), constant_excitation, dtype=f32),
        api_action=np.zeros((T, action_dim), dtype=f32),
        last_ctrl=np.zeros((T, action_dim), dtype=f32),
        reward=np.zeros(T, dtype=f32),
        terminated=np.zeros(T, dtype=np.bool_),
        truncated=np.zeros(T, dtype=np.bool_),
    )


# --- aggregate_reaching ---


def test_empty_input_returns_n_zero() -> None:
    out = aggregate_reaching([])
    assert out == {"n": 0}


def test_single_log_keys_present() -> None:
    out = aggregate_reaching([_log()])
    expected_keys = {
        "n",
        "minimum_tip_error_mean",
        "minimum_tip_error_median",
        "minimum_tip_error_std",
        "final_tip_error_mean",
        "final_tip_error_median",
        "final_tip_error_std",
        "success_rate",
        "effort_mean",
        "effort_std",
        "threshold",
        "duration",
    }
    assert set(out.keys()) == expected_keys
    assert out["n"] == 1


def test_all_zero_logs_succeed() -> None:
    logs = [_log(n_steps=20, constant_reach_dist=0.0) for _ in range(3)]
    out = aggregate_reaching(logs)
    assert out["n"] == 3
    assert out["success_rate"] == pytest.approx(1.0)
    assert out["minimum_tip_error_mean"] == 0.0
    assert out["final_tip_error_mean"] == 0.0


def test_all_far_logs_fail() -> None:
    logs = [_log(n_steps=20, constant_reach_dist=1.0) for _ in range(3)]
    out = aggregate_reaching(logs)
    assert out["success_rate"] == 0.0
    assert out["minimum_tip_error_mean"] == pytest.approx(1.0)
    assert out["final_tip_error_mean"] == pytest.approx(1.0)


def test_partial_success_rate() -> None:
    # 2 succeed (dist=0), 3 fail (dist=1) → success_rate = 0.4
    logs = (
        [_log(n_steps=20, constant_reach_dist=0.0) for _ in range(2)]
        + [_log(n_steps=20, constant_reach_dist=1.0) for _ in range(3)]
    )
    out = aggregate_reaching(logs)
    assert out["n"] == 5
    assert out["success_rate"] == pytest.approx(0.4)


def test_effort_aggregation() -> None:
    # excitation = 0.5 across all 4 muscles → ||u||² = 1.0 per step → mean = 1.0
    logs = [_log(n_steps=20, constant_excitation=0.5) for _ in range(3)]
    out = aggregate_reaching(logs)
    assert out["effort_mean"] == pytest.approx(1.0)
    assert out["effort_std"] == pytest.approx(0.0)


def test_threshold_duration_propagated() -> None:
    out = aggregate_reaching([_log()], threshold=0.1, duration=5)
    assert out["threshold"] == pytest.approx(0.1)
    assert out["duration"] == 5


def test_distance_distribution_stats() -> None:
    # Distances 0, 1, 2 → mean=1, median=1, std=sqrt(2/3)
    logs = [
        _log(n_steps=20, constant_reach_dist=d) for d in (0.0, 1.0, 2.0)
    ]
    out = aggregate_reaching(logs)
    assert out["final_tip_error_mean"] == pytest.approx(1.0)
    assert out["final_tip_error_median"] == pytest.approx(1.0)
    assert out["final_tip_error_std"] == pytest.approx(np.sqrt(2 / 3))


def test_iterable_input_accepted() -> None:
    # Generator should be consumable.
    gen = (_log() for _ in range(2))
    out = aggregate_reaching(gen)
    assert out["n"] == 2


def test_threshold_change_changes_success_rate() -> None:
    log = _log(n_steps=20, constant_reach_dist=0.07)
    # threshold 0.05: 0.07 > 0.05 → fail
    assert aggregate_reaching([log], threshold=0.05)["success_rate"] == 0.0
    # threshold 0.10: 0.07 < 0.10 → success
    assert aggregate_reaching([log], threshold=0.10)["success_rate"] == 1.0
