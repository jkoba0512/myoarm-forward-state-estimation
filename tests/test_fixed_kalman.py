"""Tests for myoarm_fse.estimators.fixed_kalman (no MyoSuite required)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from myoarm_fse.data import EpisodeLog
from myoarm_fse.envs.state import StateSpec
from myoarm_fse.estimators import (
    Estimator,
    EstimationResult,
    FixedGainKalmanEstimator,
    aggregate_estimation_metrics,
    evaluate_estimator_on_log,
    synth_observations,
)
from myoarm_fse.models import ForwardMLP


# --- helpers ---


def _make_spec(qpos_dim: int = 2, qvel_dim: int = 2, act_dim: int = 3) -> StateSpec:
    return StateSpec(qpos_dim=qpos_dim, qvel_dim=qvel_dim, act_dim=act_dim)


def _zero_forward_model(state_dim: int, action_dim: int) -> ForwardMLP:
    """ForwardMLP with all parameters zeroed → predicts Δx = 0 always."""
    m = ForwardMLP(state_dim=state_dim, action_dim=action_dim, hidden_dims=(8,))
    with torch.no_grad():
        for p in m.parameters():
            p.zero_()
    m.eval()
    return m


def _make_log(
    *,
    episode_id: int = 0,
    n_steps: int = 8,
    qpos_dim: int = 2,
    qvel_dim: int = 2,
    act_dim: int = 3,
    action_dim: int = 3,
) -> EpisodeLog:
    T = n_steps
    f32 = np.float32
    qpos = np.zeros((T, qpos_dim), dtype=f32)
    qpos[:, 0] = np.arange(T) * 0.01
    excitation = np.full((T, action_dim), 0.5, dtype=f32)
    return EpisodeLog(
        episode_id=episode_id,
        target_id=f"train:{episode_id}",
        target_split="train",
        target_seed=episode_id,
        target_pos_set=np.zeros(3, dtype=f32),
        controller_name="random",
        controller_seed=0,
        sdn_sigma=0.0,
        sdn_seed=0,
        obs_noise_sigma={},
        obs_noise_seed=0,
        obs_delay_steps=0,
        obs_compose="noisy_then_delayed",
        max_steps=T,
        n_steps=T,
        created_at="2026-05-10T00:00:00Z",
        config_hash="test",
        step=np.arange(T, dtype=np.int64),
        time=np.arange(T, dtype=f32) * 0.02,
        true_qpos=qpos,
        true_qvel=np.zeros((T, qvel_dim), dtype=f32),
        true_act=np.zeros((T, act_dim), dtype=f32),
        true_tip_pos=np.zeros((T, 3), dtype=f32),
        true_target_pos=np.zeros((T, 3), dtype=f32),
        true_reach_err=np.zeros((T, 3), dtype=f32),
        obs_qpos=qpos.copy(),
        obs_qvel=np.zeros((T, qvel_dim), dtype=f32),
        obs_act=np.zeros((T, act_dim), dtype=f32),
        obs_tip_pos=np.zeros((T, 3), dtype=f32),
        obs_target_pos=np.zeros((T, 3), dtype=f32),
        obs_reach_err=np.zeros((T, 3), dtype=f32),
        neural_command=excitation.copy(),
        excitation_command=excitation.copy(),
        excitation=excitation,
        api_action=np.zeros((T, action_dim), dtype=f32),
        last_ctrl=np.zeros((T, action_dim), dtype=f32),
        reward=np.zeros(T, dtype=f32),
        terminated=np.zeros(T, dtype=np.bool_),
        truncated=np.zeros(T, dtype=np.bool_),
    )


# --- constructor / gain validation ---


class TestConstructor:
    def test_valid_scalar_gain(self) -> None:
        spec = _make_spec()
        m = _zero_forward_model(spec.dim, action_dim=3)
        est = FixedGainKalmanEstimator(m, gain=0.5, state_spec=spec)
        assert est.state_dim == spec.dim
        assert est.action_dim == 3
        assert est.delay_steps == 0
        assert np.all(est.gain_vec == 0.5)

    def test_valid_dict_gain(self) -> None:
        spec = _make_spec()
        m = _zero_forward_model(spec.dim, action_dim=3)
        est = FixedGainKalmanEstimator(
            m, gain={"qpos": 0.5, "tip_pos": 0.8}, state_spec=spec,
        )
        layout = spec.layout()
        # qpos slice = 0.5, tip_pos slice = 0.8, others = 0
        np.testing.assert_allclose(est.gain_vec[layout["qpos"]], 0.5, atol=1e-6)
        np.testing.assert_allclose(est.gain_vec[layout["tip_pos"]], 0.8, atol=1e-6)
        np.testing.assert_allclose(est.gain_vec[layout["qvel"]], 0.0, atol=1e-6)
        np.testing.assert_allclose(est.gain_vec[layout["target_pos"]], 0.0, atol=1e-6)

    @pytest.mark.parametrize("bad", [-0.1, 1.1])
    def test_scalar_gain_out_of_range(self, bad: float) -> None:
        spec = _make_spec()
        m = _zero_forward_model(spec.dim, action_dim=3)
        with pytest.raises(ValueError, match=r"\[0.0, 1.0\]"):
            FixedGainKalmanEstimator(m, gain=bad, state_spec=spec)

    def test_bool_gain(self) -> None:
        spec = _make_spec()
        m = _zero_forward_model(spec.dim, action_dim=3)
        with pytest.raises(ValueError):
            FixedGainKalmanEstimator(m, gain=True, state_spec=spec)  # type: ignore[arg-type]

    def test_unknown_field_in_gain_dict(self) -> None:
        spec = _make_spec()
        m = _zero_forward_model(spec.dim, action_dim=3)
        with pytest.raises(ValueError, match="unknown gain field"):
            FixedGainKalmanEstimator(
                m, gain={"banana": 0.5}, state_spec=spec,
            )

    def test_per_field_gain_out_of_range(self) -> None:
        spec = _make_spec()
        m = _zero_forward_model(spec.dim, action_dim=3)
        with pytest.raises(ValueError):
            FixedGainKalmanEstimator(
                m, gain={"qpos": 1.5}, state_spec=spec,
            )

    def test_state_spec_mismatch(self) -> None:
        small = StateSpec(qpos_dim=2, qvel_dim=2, act_dim=2)
        large = StateSpec(qpos_dim=3, qvel_dim=3, act_dim=4)
        m = _zero_forward_model(small.dim, action_dim=3)
        with pytest.raises(ValueError, match="state_dim"):
            FixedGainKalmanEstimator(m, gain=0.5, state_spec=large)

    def test_negative_delay(self) -> None:
        spec = _make_spec()
        m = _zero_forward_model(spec.dim, action_dim=3)
        with pytest.raises(ValueError):
            FixedGainKalmanEstimator(m, gain=0.5, state_spec=spec, delay_steps=-1)

    def test_bool_delay(self) -> None:
        spec = _make_spec()
        m = _zero_forward_model(spec.dim, action_dim=3)
        with pytest.raises(ValueError):
            FixedGainKalmanEstimator(m, gain=0.5, state_spec=spec, delay_steps=True)  # type: ignore[arg-type]

    def test_protocol_conformance(self) -> None:
        spec = _make_spec()
        m = _zero_forward_model(spec.dim, action_dim=3)
        est = FixedGainKalmanEstimator(m, gain=0.5, state_spec=spec)
        assert isinstance(est, Estimator)


# --- step semantics with zero forward model (Δx = 0) ---


class TestStepSemanticsZeroModel:
    def test_k_zero_returns_initial_state_when_model_predicts_zero(self) -> None:
        """K=0 → prediction-only. Zero model → x_est stays at initial_state."""
        spec = _make_spec()
        m = _zero_forward_model(spec.dim, action_dim=3)
        est = FixedGainKalmanEstimator(m, gain=0.0, state_spec=spec, delay_steps=0)
        init = np.full(spec.dim, 1.5, dtype=np.float32)
        est.reset(init)
        for _ in range(5):
            out = est.step(
                y_obs=np.zeros(spec.dim, dtype=np.float32),
                u=np.full(3, 0.5, dtype=np.float32),
            )
            np.testing.assert_array_equal(out, init)

    def test_k_one_jumps_to_observation(self) -> None:
        """K=1 → x_est = y_obs (observation-only) when model predicts zero."""
        spec = _make_spec()
        m = _zero_forward_model(spec.dim, action_dim=3)
        est = FixedGainKalmanEstimator(m, gain=1.0, state_spec=spec, delay_steps=0)
        init = np.zeros(spec.dim, dtype=np.float32)
        est.reset(init)
        target = np.full(spec.dim, 2.5, dtype=np.float32)
        out = est.step(y_obs=target, u=np.zeros(3, dtype=np.float32))
        np.testing.assert_allclose(out, target, atol=1e-6)

    def test_k_half_blends(self) -> None:
        """K=0.5 → average of prediction (=init for zero model) and y_obs."""
        spec = _make_spec()
        m = _zero_forward_model(spec.dim, action_dim=3)
        est = FixedGainKalmanEstimator(m, gain=0.5, state_spec=spec, delay_steps=0)
        init = np.full(spec.dim, 0.0, dtype=np.float32)
        est.reset(init)
        y = np.full(spec.dim, 2.0, dtype=np.float32)
        out = est.step(y_obs=y, u=np.zeros(3, dtype=np.float32))
        np.testing.assert_allclose(out, np.full(spec.dim, 1.0), atol=1e-6)

    def test_per_field_gain_partial_update(self) -> None:
        """Per-field K updates only the specified fields."""
        spec = _make_spec()
        m = _zero_forward_model(spec.dim, action_dim=3)
        est = FixedGainKalmanEstimator(
            m, gain={"qpos": 1.0}, state_spec=spec, delay_steps=0,
        )
        layout = spec.layout()
        init = np.zeros(spec.dim, dtype=np.float32)
        est.reset(init)
        y = np.full(spec.dim, 5.0, dtype=np.float32)
        out = est.step(y_obs=y, u=np.zeros(3, dtype=np.float32))
        # qpos slice == y, others stay at 0 (K=0 → prediction = init = 0).
        np.testing.assert_allclose(out[layout["qpos"]], 5.0, atol=1e-6)
        np.testing.assert_allclose(out[layout["qvel"]], 0.0, atol=1e-6)
        np.testing.assert_allclose(out[layout["act"]], 0.0, atol=1e-6)


# --- step before reset ---


def test_step_before_reset_raises() -> None:
    spec = _make_spec()
    m = _zero_forward_model(spec.dim, action_dim=3)
    est = FixedGainKalmanEstimator(m, gain=0.5, state_spec=spec)
    with pytest.raises(RuntimeError, match="before reset"):
        est.step(np.zeros(spec.dim, dtype=np.float32), np.zeros(3, dtype=np.float32))


# --- input validation ---


class TestInputValidation:
    def test_y_obs_wrong_shape(self) -> None:
        spec = _make_spec()
        m = _zero_forward_model(spec.dim, action_dim=3)
        est = FixedGainKalmanEstimator(m, gain=0.5, state_spec=spec)
        est.reset(np.zeros(spec.dim, dtype=np.float32))
        with pytest.raises(ValueError, match="y_obs"):
            est.step(np.zeros(spec.dim - 1, dtype=np.float32),
                     np.zeros(3, dtype=np.float32))

    def test_u_wrong_shape(self) -> None:
        spec = _make_spec()
        m = _zero_forward_model(spec.dim, action_dim=3)
        est = FixedGainKalmanEstimator(m, gain=0.5, state_spec=spec)
        est.reset(np.zeros(spec.dim, dtype=np.float32))
        with pytest.raises(ValueError, match="u"):
            est.step(np.zeros(spec.dim, dtype=np.float32),
                     np.zeros(2, dtype=np.float32))

    def test_initial_state_wrong_shape(self) -> None:
        spec = _make_spec()
        m = _zero_forward_model(spec.dim, action_dim=3)
        est = FixedGainKalmanEstimator(m, gain=0.5, state_spec=spec)
        with pytest.raises(ValueError, match="initial_state"):
            est.reset(np.zeros(spec.dim - 1, dtype=np.float32))


# --- delay handling ---


class TestDelayHandling:
    def test_cold_start_returns_prediction_only(self) -> None:
        """During cold start (t < delay_steps), prediction is propagated."""
        spec = _make_spec()
        m = _zero_forward_model(spec.dim, action_dim=3)
        est = FixedGainKalmanEstimator(
            m, gain=1.0, state_spec=spec, delay_steps=2,
        )
        init = np.zeros(spec.dim, dtype=np.float32)
        est.reset(init)
        # First two steps: u_buffer cannot accumulate enough actions, so
        # estimator should NOT correct toward y_obs.
        for _ in range(2):
            y = np.full(spec.dim, 99.0, dtype=np.float32)
            out = est.step(y, np.zeros(3, dtype=np.float32))
            # Zero model + cold start → returns init (no correction applied).
            np.testing.assert_allclose(out, init, atol=1e-6)

    def test_after_cold_start_correction_applies(self) -> None:
        spec = _make_spec()
        m = _zero_forward_model(spec.dim, action_dim=3)
        est = FixedGainKalmanEstimator(
            m, gain=1.0, state_spec=spec, delay_steps=2,
        )
        init = np.zeros(spec.dim, dtype=np.float32)
        est.reset(init)
        # Two cold-start steps
        for _ in range(2):
            est.step(np.zeros(spec.dim, dtype=np.float32),
                     np.zeros(3, dtype=np.float32))
        # Third step has full delay buffer; gain=1 + zero model → correction
        # rolls the past correction forward (still zeros so no change either,
        # but no longer in cold start). Confirm by sending a non-zero y_obs.
        y = np.full(spec.dim, 3.0, dtype=np.float32)
        out = est.step(y, np.zeros(3, dtype=np.float32))
        # gain=1, model=zero → corrected past = y, rolled forward 2 steps with
        # zero delta = y still. So out should equal y.
        np.testing.assert_allclose(out, y, atol=1e-6)

    def test_delay_zero_path(self) -> None:
        spec = _make_spec()
        m = _zero_forward_model(spec.dim, action_dim=3)
        est0 = FixedGainKalmanEstimator(m, gain=0.5, state_spec=spec, delay_steps=0)
        est0.reset(np.zeros(spec.dim, dtype=np.float32))
        out0 = est0.step(
            np.full(spec.dim, 4.0, dtype=np.float32),
            np.zeros(3, dtype=np.float32),
        )
        # Zero model: x_pred=init=0, K=0.5, y=4 → x_est = 0 + 0.5*(4-0) = 2.0.
        np.testing.assert_allclose(out0, np.full(spec.dim, 2.0), atol=1e-6)


# --- synth_observations ---


class TestSynthObservations:
    def test_zero_noise_zero_delay_matches_true(self) -> None:
        spec = _make_spec()
        log = _make_log(qpos_dim=2, qvel_dim=2, act_dim=3, action_dim=3)
        y_obs = synth_observations(
            log, state_spec=spec, sigma={}, delay_steps=0, seed=0,
        )
        # With no noise and no delay, y_obs equals flatten(true_state).
        from myoarm_fse.estimators.fixed_kalman import _flatten_log_states
        x_true = _flatten_log_states(log)
        np.testing.assert_allclose(y_obs, x_true, atol=1e-6)

    def test_delay_shifts_observations(self) -> None:
        spec = _make_spec()
        log = _make_log()
        y_obs = synth_observations(
            log, state_spec=spec, sigma={}, delay_steps=2, seed=0,
        )
        from myoarm_fse.estimators.fixed_kalman import _flatten_log_states
        x_true = _flatten_log_states(log)
        # The first delay_steps observations equal the initial state
        # (delay buffer init).
        for t in range(2):
            np.testing.assert_allclose(y_obs[t], x_true[0], atol=1e-6)
        # After cold start, y_obs[t] = x_true[t-2].
        for t in range(2, log.n_steps):
            np.testing.assert_allclose(y_obs[t], x_true[t - 2], atol=1e-6)

    def test_noise_is_per_field(self) -> None:
        spec = _make_spec()
        log = _make_log()
        y_obs = synth_observations(
            log, state_spec=spec, sigma={"qpos": 0.5}, delay_steps=0, seed=0,
        )
        from myoarm_fse.estimators.fixed_kalman import _flatten_log_states
        x_true = _flatten_log_states(log)
        layout = spec.layout()
        # qpos slice should differ; qvel / act should be exactly equal.
        assert not np.allclose(y_obs[:, layout["qpos"]], x_true[:, layout["qpos"]])
        np.testing.assert_allclose(y_obs[:, layout["qvel"]],
                                   x_true[:, layout["qvel"]], atol=1e-6)
        np.testing.assert_allclose(y_obs[:, layout["act"]],
                                   x_true[:, layout["act"]], atol=1e-6)

    def test_invalid_obs_compose(self) -> None:
        spec = _make_spec()
        log = _make_log()
        with pytest.raises(ValueError, match="obs_compose"):
            synth_observations(
                log, state_spec=spec, sigma={}, delay_steps=0, seed=0,
                obs_compose="banana",
            )


# --- evaluate_estimator_on_log ---


class TestEvaluateEstimatorOnLog:
    def test_zero_error_oracle_path(self) -> None:
        """K=1, delay=0, sigma={} → x_est should track x_true exactly."""
        spec = _make_spec()
        m = _zero_forward_model(spec.dim, action_dim=3)
        est = FixedGainKalmanEstimator(m, gain=1.0, state_spec=spec, delay_steps=0)
        log = _make_log()
        result = evaluate_estimator_on_log(
            est, log,
            state_spec=spec,
            obs_noise_sigma={},
            obs_delay_steps=0,
            obs_noise_seed=0,
        )
        assert isinstance(result, EstimationResult)
        assert result.n_steps == log.n_steps
        assert result.delay_steps == 0
        np.testing.assert_allclose(
            result.error_per_step, 0.0, atol=1e-5,
        )
        np.testing.assert_allclose(result.error_per_step_norm, 0.0, atol=1e-5)

    def test_shape_and_layout(self) -> None:
        spec = _make_spec()
        m = _zero_forward_model(spec.dim, action_dim=3)
        est = FixedGainKalmanEstimator(m, gain=0.5, state_spec=spec)
        log = _make_log()
        result = evaluate_estimator_on_log(
            est, log,
            state_spec=spec,
            obs_noise_sigma={"qpos": 0.1},
            obs_delay_steps=0,
            obs_noise_seed=0,
        )
        assert result.x_est.shape == (log.n_steps, spec.dim)
        assert result.x_true.shape == (log.n_steps, spec.dim)
        assert result.error_per_step.shape == (log.n_steps, spec.dim)
        assert set(result.layout.keys()) == {
            "qpos", "qvel", "act", "tip_pos", "target_pos", "reach_err"
        }


# --- aggregate_estimation_metrics ---


class TestAggregateEstimationMetrics:
    def test_empty_input(self) -> None:
        out = aggregate_estimation_metrics([])
        assert out == {"n": 0}

    def test_zero_error_aggregates_to_zero(self) -> None:
        spec = _make_spec()
        m = _zero_forward_model(spec.dim, action_dim=3)
        est = FixedGainKalmanEstimator(m, gain=1.0, state_spec=spec)
        log = _make_log()
        result = evaluate_estimator_on_log(
            est, log,
            state_spec=spec,
            obs_noise_sigma={},
            obs_delay_steps=0,
            obs_noise_seed=0,
        )
        out = aggregate_estimation_metrics([result])
        assert out["n"] == 1
        for name in ("qpos", "qvel", "act", "tip_pos", "target_pos", "reach_err"):
            assert out[f"mse_{name}_mean"] == pytest.approx(0.0, abs=1e-6)
        assert out["tip_estimation_error_mean"] == pytest.approx(0.0, abs=1e-6)

    def test_skip_cold_start_drops_short_episodes(self) -> None:
        spec = _make_spec()
        m = _zero_forward_model(spec.dim, action_dim=3)
        est = FixedGainKalmanEstimator(m, gain=0.5, state_spec=spec, delay_steps=10)
        log = _make_log(n_steps=5)  # shorter than delay_steps=10
        result = evaluate_estimator_on_log(
            est, log,
            state_spec=spec,
            obs_noise_sigma={},
            obs_delay_steps=10,
            obs_noise_seed=0,
        )
        out = aggregate_estimation_metrics([result], skip_cold_start=True)
        # Episode is fully cold-start → contributes nothing.
        assert out["n"] == 0
