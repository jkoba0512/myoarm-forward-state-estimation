"""Tests for Stage B state-aware gain predictor and per-step estimator."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from myoarm_fse.envs.state import StateSpec
from myoarm_fse.estimators import (
    Estimator,
    GainPredictor,
    StateAwareGainPredictor,
    StateAwareLearnedGainKalmanEstimator,
    load_learned_gain_model,
    save_learned_gain_model,
)
from myoarm_fse.estimators.learned import (
    _encode_state_features,
    state_aware_feature_dim,
)
from myoarm_fse.models import ForwardMLP


def _spec() -> StateSpec:
    return StateSpec(qpos_dim=2, qvel_dim=2, act_dim=3)


def _zero_forward_model(state_dim: int, action_dim: int) -> ForwardMLP:
    m = ForwardMLP(state_dim=state_dim, action_dim=action_dim, hidden_dims=(8,))
    with torch.no_grad():
        for p in m.parameters():
            p.zero_()
    m.eval()
    return m


# --- _encode_state_features ---


class TestEncodeStateFeatures:
    def test_per_field_norms(self) -> None:
        spec = _spec()
        # innovation = ones in all slots
        inn = np.ones(spec.dim, dtype=np.float32)
        out = _encode_state_features(inn, state_spec=spec)
        assert out.shape == (4,)
        assert out.dtype == np.float32
        # qpos 2-dim → sqrt(2), qvel 2-dim → sqrt(2),
        # tip_pos 3-dim → sqrt(3), reach_err 3-dim → sqrt(3).
        np.testing.assert_allclose(
            out, [np.sqrt(2), np.sqrt(2), np.sqrt(3), np.sqrt(3)], rtol=1e-6,
        )

    def test_zero_innovation_yields_zero(self) -> None:
        spec = _spec()
        inn = np.zeros(spec.dim, dtype=np.float32)
        out = _encode_state_features(inn, state_spec=spec)
        np.testing.assert_array_equal(out, np.zeros(4, dtype=np.float32))

    def test_shape_mismatch_raises(self) -> None:
        spec = _spec()
        bad = np.zeros(spec.dim + 1, dtype=np.float32)
        with pytest.raises(ValueError, match="must have shape"):
            _encode_state_features(bad, state_spec=spec)

    def test_unknown_field_raises(self) -> None:
        spec = _spec()
        inn = np.zeros(spec.dim, dtype=np.float32)
        with pytest.raises(ValueError, match="unknown state_feature_fields"):
            _encode_state_features(
                inn, state_spec=spec, state_feature_fields=("qpos", "banana"),
            )


def test_state_aware_feature_dim_default() -> None:
    assert state_aware_feature_dim() == 12  # 3 + 4 + 1 + 4


# --- StateAwareGainPredictor ---


class TestStateAwareGainPredictor:
    def test_default_construction(self) -> None:
        m = StateAwareGainPredictor()
        assert m.in_dim == 12
        assert m.hidden_dims == (64, 64)
        # Param count: 12*64 + 64 + 64*64 + 64 + 64*1 + 1
        # = 768 + 64 + 4096 + 64 + 64 + 1 = 5057
        assert m.num_parameters() > 5000

    def test_forward_shape_and_range(self) -> None:
        m = StateAwareGainPredictor()
        x = torch.zeros((4, 12))
        out = m(x)
        assert out.shape == (4,)
        assert (out >= 0).all() and (out <= 1).all()

    def test_forward_single_sample(self) -> None:
        m = StateAwareGainPredictor()
        x = torch.zeros(12)
        out = m(x)
        assert out.shape == ()

    def test_input_dim_mismatch(self) -> None:
        m = StateAwareGainPredictor()
        with pytest.raises(ValueError, match="x.shape"):
            m(torch.zeros((3, 10)))

    def test_invalid_hidden_dims(self) -> None:
        with pytest.raises(ValueError):
            StateAwareGainPredictor(hidden_dims=(64, 0))

    def test_invalid_n_state_features(self) -> None:
        with pytest.raises(ValueError, match="n_state_features"):
            StateAwareGainPredictor(n_state_features=0)

    def test_init_deterministic_with_torch_seed(self) -> None:
        torch.manual_seed(0)
        a = StateAwareGainPredictor()
        torch.manual_seed(0)
        b = StateAwareGainPredictor()
        for (_, pa), (_, pb) in zip(a.named_parameters(), b.named_parameters()):
            torch.testing.assert_close(pa, pb)

    def test_gradient_flows(self) -> None:
        m = StateAwareGainPredictor()
        x = torch.randn((3, 12))
        target = torch.zeros(3)
        loss = ((m(x) - target) ** 2).mean()
        loss.backward()
        for name, p in m.named_parameters():
            assert p.grad is not None, f"no grad for {name}"
            assert torch.isfinite(p.grad).all()


# --- StateAwareLearnedGainKalmanEstimator ---


class TestStateAwareLearnedGainKalmanEstimator:
    def test_protocol_conformance(self) -> None:
        spec = _spec()
        fm = _zero_forward_model(spec.dim, action_dim=3)
        predictor = StateAwareGainPredictor()
        est = StateAwareLearnedGainKalmanEstimator(
            forward_model=fm,
            gain_predictor=predictor,
            state_spec=spec,
            delay_steps=0,
            controller_name="random",
            noise_sigma={"qpos": 0.01},
        )
        assert isinstance(est, Estimator)
        assert est.state_dim == spec.dim
        assert est.controller_name == "random"

    def test_unknown_controller_raises(self) -> None:
        spec = _spec()
        fm = _zero_forward_model(spec.dim, action_dim=3)
        predictor = StateAwareGainPredictor()
        with pytest.raises(ValueError, match="not in known"):
            StateAwareLearnedGainKalmanEstimator(
                forward_model=fm,
                gain_predictor=predictor,
                state_spec=spec,
                controller_name="banana",
                noise_sigma={},
            )

    def test_non_predictor_input(self) -> None:
        spec = _spec()
        fm = _zero_forward_model(spec.dim, action_dim=3)
        with pytest.raises(TypeError):
            StateAwareLearnedGainKalmanEstimator(
                forward_model=fm,
                gain_predictor="not a predictor",  # type: ignore[arg-type]
                state_spec=spec,
                controller_name="hold",
                noise_sigma={},
            )

    def test_step_after_reset(self) -> None:
        spec = _spec()
        fm = _zero_forward_model(spec.dim, action_dim=3)
        predictor = StateAwareGainPredictor()
        est = StateAwareLearnedGainKalmanEstimator(
            forward_model=fm, gain_predictor=predictor, state_spec=spec,
            delay_steps=0, controller_name="hold", noise_sigma={},
        )
        est.reset(np.zeros(spec.dim, dtype=np.float32))
        # zero forward + init=0 → x_pred=0, innovation = y
        y = np.full(spec.dim, 0.5, dtype=np.float32)
        out = est.step(y, np.zeros(3, dtype=np.float32))
        assert out.shape == (spec.dim,)
        # K must be in [0, 1] so out = K * y, hence each entry in [0, 0.5].
        assert np.all(out >= 0.0)
        assert np.all(out <= 0.5)
        assert len(est.k_history) == 1
        assert 0.0 <= est.k_history[0] <= 1.0

    def test_reset_required_before_step(self) -> None:
        spec = _spec()
        fm = _zero_forward_model(spec.dim, action_dim=3)
        predictor = StateAwareGainPredictor()
        est = StateAwareLearnedGainKalmanEstimator(
            forward_model=fm, gain_predictor=predictor, state_spec=spec,
            delay_steps=0, controller_name="hold", noise_sigma={},
        )
        with pytest.raises(RuntimeError):
            est.step(
                np.zeros(spec.dim, dtype=np.float32),
                np.zeros(3, dtype=np.float32),
            )

    def test_k_history_varies_with_innovation(self) -> None:
        """Different innovation should generally produce different K predictions."""
        spec = _spec()
        fm = _zero_forward_model(spec.dim, action_dim=3)
        predictor = StateAwareGainPredictor()
        # Force a non-trivial weight pattern so K depends on input.
        with torch.no_grad():
            for p in predictor.parameters():
                p.mul_(0.0).add_(torch.randn_like(p) * 0.5)
        est = StateAwareLearnedGainKalmanEstimator(
            forward_model=fm, gain_predictor=predictor, state_spec=spec,
            delay_steps=0, controller_name="hold", noise_sigma={},
        )
        est.reset(np.zeros(spec.dim, dtype=np.float32))
        out_a = est.step(np.zeros(spec.dim, dtype=np.float32),
                         np.zeros(3, dtype=np.float32))
        out_b = est.step(np.full(spec.dim, 10.0, dtype=np.float32),
                         np.zeros(3, dtype=np.float32))
        # K values should differ at zero vs large innovation (with random weights).
        assert est.k_history[0] != est.k_history[1]

    def test_delay_cold_start_uses_prediction(self) -> None:
        spec = _spec()
        fm = _zero_forward_model(spec.dim, action_dim=3)
        predictor = StateAwareGainPredictor()
        est = StateAwareLearnedGainKalmanEstimator(
            forward_model=fm, gain_predictor=predictor, state_spec=spec,
            delay_steps=2, controller_name="hold", noise_sigma={},
        )
        est.reset(np.full(spec.dim, 0.3, dtype=np.float32))
        # First step is cold-start (no past actions yet).
        out = est.step(np.full(spec.dim, 0.9, dtype=np.float32),
                       np.zeros(3, dtype=np.float32))
        # zero forward + init=0.3 → x_pred = 0.3; no correction → out = 0.3
        np.testing.assert_allclose(out, 0.3, rtol=1e-5)
        # K history empty (no correction made during cold start)
        assert est.k_history == []


# --- save / load round trip with StateAwareGainPredictor ---


class TestStateAwareSaveLoad:
    def test_round_trip(self, tmp_path: Path) -> None:
        m = StateAwareGainPredictor(hidden_dims=(32, 16))
        config = {
            "architecture": {
                "kind": "StateAwareGainPredictor",
                "n_controllers": 3,
                "n_sigma_fields": 4,
                "n_state_features": 4,
                "hidden_dims": [32, 16],
            },
            "train": {"epochs": 10},
        }
        metrics = {"final_train_loss": 0.01}
        path = tmp_path / "stage_b_model"
        save_learned_gain_model(m, config, metrics, path=path)
        loaded_model, loaded_config, loaded_metrics = load_learned_gain_model(path)
        assert isinstance(loaded_model, StateAwareGainPredictor)
        assert loaded_model.hidden_dims == (32, 16)
        for (_, pa), (_, pb) in zip(m.named_parameters(),
                                     loaded_model.named_parameters()):
            torch.testing.assert_close(pa, pb)
        assert loaded_config == config

    def test_load_dispatches_on_kind(self, tmp_path: Path) -> None:
        # Save a Stage A model (kind=GainPredictor) and ensure load
        # returns GainPredictor, not StateAwareGainPredictor.
        a = GainPredictor()
        config = {
            "architecture": {
                "kind": "GainPredictor",
                "n_controllers": 3,
                "n_sigma_fields": 4,
                "hidden_dims": [32, 32],
            },
            "train": {"epochs": 10},
        }
        path = tmp_path / "stage_a_model"
        save_learned_gain_model(a, config, {}, path=path)
        loaded, _, _ = load_learned_gain_model(path)
        assert isinstance(loaded, GainPredictor)
        assert not isinstance(loaded, StateAwareGainPredictor)

    def test_unknown_kind_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "bad"
        path.mkdir()
        (path / "config.json").write_text(
            '{"architecture": {"kind": "Banana"}}'
        )
        # Need a model.pt and metrics.json to even get past those steps;
        # but kind check happens before we load the state_dict.
        (path / "metrics.json").write_text("{}")
        (path / "model.pt").write_bytes(b"")  # bogus, won't be reached
        with pytest.raises(ValueError, match="unknown architecture.kind"):
            load_learned_gain_model(path)
