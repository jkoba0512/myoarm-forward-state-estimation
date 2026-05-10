"""Tests for myoarm_fse.estimators.learned (no MyoSuite required)."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
import torch

from myoarm_fse.envs.state import StateSpec
from myoarm_fse.estimators import (
    Estimator,
    GainPredictor,
    LearnedGainKalmanEstimator,
    load_learned_gain_model,
    load_oracle_table,
    save_learned_gain_model,
)
from myoarm_fse.estimators.learned import _encode_features, feature_dim
from myoarm_fse.models import ForwardMLP


# --- _encode_features ---


class TestEncodeFeatures:
    def test_basic_shape(self) -> None:
        out = _encode_features(
            controller="random",
            noise_sigma={"qpos": 0.01, "qvel": 0.05, "tip_pos": 0.005, "reach_err": 0.005},
            delay_steps=2,
        )
        assert out.shape == (8,)
        assert out.dtype == np.float32

    def test_controller_one_hot(self) -> None:
        out = _encode_features(
            controller="lowamp",
            noise_sigma={},
            delay_steps=0,
        )
        # default controller_names = (random, lowamp, hold) → idx 1
        assert out[0] == 0.0
        assert out[1] == 1.0
        assert out[2] == 0.0

    def test_unknown_controller_yields_zero_one_hot(self) -> None:
        out = _encode_features(
            controller="banana",
            noise_sigma={},
            delay_steps=0,
        )
        assert out[0] == 0.0 and out[1] == 0.0 and out[2] == 0.0

    def test_sigma_vector_in_field_order(self) -> None:
        out = _encode_features(
            controller="hold",
            noise_sigma={"qpos": 0.01, "qvel": 0.02, "tip_pos": 0.03, "reach_err": 0.04},
            delay_steps=0,
        )
        # default sigma_field_order = (qpos, qvel, tip_pos, reach_err) → indices 3..6
        assert out[3] == pytest.approx(0.01)
        assert out[4] == pytest.approx(0.02)
        assert out[5] == pytest.approx(0.03)
        assert out[6] == pytest.approx(0.04)

    def test_missing_sigma_field_defaults_zero(self) -> None:
        out = _encode_features(
            controller="hold",
            noise_sigma={"qpos": 0.01},
            delay_steps=0,
        )
        assert out[3] == pytest.approx(0.01)
        assert out[4] == 0.0
        assert out[5] == 0.0
        assert out[6] == 0.0

    def test_delay_normalized(self) -> None:
        out = _encode_features(
            controller="hold",
            noise_sigma={},
            delay_steps=3,
            delay_max=6,
        )
        assert out[7] == pytest.approx(0.5)

    def test_zero_delay_max_raises(self) -> None:
        with pytest.raises(ValueError):
            _encode_features(
                controller="hold",
                noise_sigma={},
                delay_steps=0,
                delay_max=0,
            )


def test_feature_dim_default() -> None:
    assert feature_dim() == 8


# --- GainPredictor ---


class TestGainPredictor:
    def test_default_construction(self) -> None:
        m = GainPredictor()
        assert m.in_dim == 8
        assert m.hidden_dims == (32, 32)
        # 8→32→32→1 expected param count: 8*32+32 + 32*32+32 + 32*1+1 = 1409
        assert m.num_parameters() > 1000
        assert m.num_parameters() < 2000

    def test_forward_shape_and_range(self) -> None:
        m = GainPredictor()
        x = torch.zeros((5, 8))
        out = m(x)
        assert out.shape == (5,)
        assert (out >= 0).all() and (out <= 1).all()

    def test_forward_single(self) -> None:
        m = GainPredictor()
        x = torch.zeros(8)
        out = m(x)
        assert out.shape == ()

    def test_input_dim_mismatch(self) -> None:
        m = GainPredictor()
        with pytest.raises(ValueError, match="x.shape"):
            m(torch.zeros((3, 7)))

    def test_gradient_flows(self) -> None:
        m = GainPredictor()
        x = torch.randn((4, 8))
        target = torch.zeros(4)
        out = m(x)
        loss = ((out - target) ** 2).mean()
        loss.backward()
        for name, p in m.named_parameters():
            assert p.grad is not None, f"no grad for {name}"
            assert torch.isfinite(p.grad).all()

    def test_invalid_hidden_dims(self) -> None:
        with pytest.raises(ValueError):
            GainPredictor(hidden_dims=(32, 0))

    def test_init_deterministic_with_torch_seed(self) -> None:
        torch.manual_seed(0)
        a = GainPredictor()
        torch.manual_seed(0)
        b = GainPredictor()
        for (_, pa), (_, pb) in zip(a.named_parameters(), b.named_parameters()):
            torch.testing.assert_close(pa, pb)


# --- LearnedGainKalmanEstimator ---


def _zero_forward_model(state_dim: int, action_dim: int) -> ForwardMLP:
    m = ForwardMLP(state_dim=state_dim, action_dim=action_dim, hidden_dims=(8,))
    with torch.no_grad():
        for p in m.parameters():
            p.zero_()
    m.eval()
    return m


def _spec() -> StateSpec:
    return StateSpec(qpos_dim=2, qvel_dim=2, act_dim=3)


class TestLearnedGainKalmanEstimator:
    def test_protocol_conformance(self) -> None:
        spec = _spec()
        m = _zero_forward_model(spec.dim, action_dim=3)
        predictor = GainPredictor()
        est = LearnedGainKalmanEstimator(
            forward_model=m,
            gain_predictor=predictor,
            state_spec=spec,
            controller_name="random",
            noise_sigma={"qpos": 0.01},
            delay_steps=0,
        )
        assert isinstance(est, Estimator)
        assert 0.0 <= est.learned_k <= 1.0
        assert est.state_dim == spec.dim
        assert est.controller_name == "random"

    def test_unknown_controller_raises(self) -> None:
        spec = _spec()
        m = _zero_forward_model(spec.dim, action_dim=3)
        predictor = GainPredictor()
        with pytest.raises(ValueError, match="not in known"):
            LearnedGainKalmanEstimator(
                forward_model=m,
                gain_predictor=predictor,
                state_spec=spec,
                controller_name="banana",
                noise_sigma={},
            )

    def test_step_after_reset(self) -> None:
        spec = _spec()
        m = _zero_forward_model(spec.dim, action_dim=3)
        predictor = GainPredictor()
        est = LearnedGainKalmanEstimator(
            forward_model=m, gain_predictor=predictor, state_spec=spec,
            controller_name="hold", noise_sigma={}, delay_steps=0,
        )
        init = np.zeros(spec.dim, dtype=np.float32)
        est.reset(init)
        out = est.step(
            np.full(spec.dim, 0.7, dtype=np.float32),
            np.zeros(3, dtype=np.float32),
        )
        # zero forward model + sigmoid(unknown init) gives some K ∈ (0,1).
        # x_pred = 0 (zero model + init=0); innovation = y; corrected = K * y.
        np.testing.assert_allclose(out, est.learned_k * 0.7, atol=1e-5)

    def test_reset_required_before_step(self) -> None:
        spec = _spec()
        m = _zero_forward_model(spec.dim, action_dim=3)
        predictor = GainPredictor()
        est = LearnedGainKalmanEstimator(
            forward_model=m, gain_predictor=predictor, state_spec=spec,
            controller_name="hold", noise_sigma={}, delay_steps=0,
        )
        with pytest.raises(RuntimeError):
            est.step(
                np.zeros(spec.dim, dtype=np.float32),
                np.zeros(3, dtype=np.float32),
            )

    def test_non_predictor_input(self) -> None:
        spec = _spec()
        m = _zero_forward_model(spec.dim, action_dim=3)
        with pytest.raises(TypeError):
            LearnedGainKalmanEstimator(
                forward_model=m, gain_predictor="not a predictor",  # type: ignore[arg-type]
                state_spec=spec, controller_name="hold", noise_sigma={},
            )


# --- load_oracle_table ---


def _write_minimal_oracle_csv(path: Path) -> None:
    rows = [
        {
            "controller": "random", "noise_condition": "none",
            "delay_steps": "0", "gain": "1.0",
            "tip_estimation_error_mean": "0.001",
            "n_episodes": "2",
        },
        {
            "controller": "random", "noise_condition": "high",
            "delay_steps": "0", "gain": "0.5",
            "tip_estimation_error_mean": "0.01",
            "n_episodes": "2",
        },
    ]
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def test_load_oracle_table_parses_numerics(tmp_path: Path) -> None:
    path = tmp_path / "best.csv"
    _write_minimal_oracle_csv(path)
    rows = load_oracle_table(path)
    assert len(rows) == 2
    assert rows[0]["controller"] == "random"
    assert rows[0]["noise_condition"] == "none"
    assert rows[0]["delay_steps"] == 0
    assert rows[0]["gain"] == pytest.approx(1.0)
    assert rows[1]["gain"] == pytest.approx(0.5)


def test_load_oracle_table_empty_raises(tmp_path: Path) -> None:
    path = tmp_path / "empty.csv"
    path.write_text("controller,noise_condition,delay_steps,gain\n")
    with pytest.raises(ValueError, match="no rows"):
        load_oracle_table(path)


# --- save / load model directory ---


class TestSaveLoadLearnedGainModel:
    def test_round_trip(self, tmp_path: Path) -> None:
        m = GainPredictor()
        config = {
            "architecture": {
                "kind": "GainPredictor",
                "n_controllers": 3,
                "n_sigma_fields": 4,
                "hidden_dims": [32, 32],
            },
            "train": {"epochs": 10},
        }
        metrics = {"final_train_abs_error_mean": 0.05}
        path = tmp_path / "model_dir"
        save_learned_gain_model(m, config, metrics, path=path, info={"oracle": "x"})
        loaded_model, loaded_config, loaded_metrics = load_learned_gain_model(path)
        # parameter equality
        for (_, pa), (_, pb) in zip(m.named_parameters(), loaded_model.named_parameters()):
            torch.testing.assert_close(pa, pb)
        assert loaded_config == config
        assert loaded_metrics == metrics
        assert (path / "info.json").exists()
