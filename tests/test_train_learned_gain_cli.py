"""Tests for myoarm_fse.estimators.learned training helpers (no MyoSuite)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from myoarm_fse.estimators import (
    LearnedGainTrainConfig,
    train_gain_predictor,
)
from myoarm_fse.estimators.learned import _make_cv_folds


def _oracle_rows() -> list[dict]:
    """Build a minimal Phase 3.3-min-shaped oracle table.

    3 controllers × 2 noise levels × 2 delays = 12 rows so each
    cv strategy has at least 2 folds and abs-error stats are stable.
    """
    rows = []
    for ctrl in ("random", "lowamp", "hold"):
        for noise in ("none", "high"):
            for delay in (0, 6):
                # Synthetic oracle K rule: high noise + short delay → K=0.5,
                # otherwise K=1.0. This is what we hope the MLP learns.
                K = 0.5 if (noise == "high" and delay == 0) else 1.0
                rows.append({
                    "controller": ctrl,
                    "noise_condition": noise,
                    "delay_steps": delay,
                    "gain": K,
                })
    return rows


def _noise_conditions() -> dict[str, dict[str, float]]:
    return {
        "none": {"qpos": 0.0, "qvel": 0.0, "tip_pos": 0.0, "reach_err": 0.0},
        "high": {"qpos": 0.02, "qvel": 0.02, "tip_pos": 0.01, "reach_err": 0.01},
    }


# --- LearnedGainTrainConfig ---


class TestLearnedGainTrainConfig:
    def test_defaults(self) -> None:
        cfg = LearnedGainTrainConfig()
        assert cfg.optimizer == "adam"
        assert cfg.lr == 1e-3
        assert cfg.weight_decay == 1e-3
        assert cfg.batch_size == 36
        assert cfg.epochs == 500

    def test_invalid_optimizer(self) -> None:
        with pytest.raises(ValueError):
            LearnedGainTrainConfig(optimizer="sgd")

    def test_negative_lr(self) -> None:
        with pytest.raises(ValueError):
            LearnedGainTrainConfig(lr=-1e-3)

    def test_zero_epochs(self) -> None:
        with pytest.raises(ValueError):
            LearnedGainTrainConfig(epochs=0)

    def test_unknown_cv_strategy(self) -> None:
        with pytest.raises(ValueError, match="unknown cv_strategy"):
            LearnedGainTrainConfig(cv_strategies=("loo", "banana"))

    def test_from_dict_unknown_keys(self) -> None:
        with pytest.raises(ValueError, match="unknown LearnedGainTrainConfig"):
            LearnedGainTrainConfig.from_dict({"banana": 1})

    def test_from_dict_with_list_cv_strategies(self) -> None:
        # YAML lists become tuples internally.
        cfg = LearnedGainTrainConfig.from_dict({"cv_strategies": ["loo", "noise"]})
        assert cfg.cv_strategies == ("loo", "noise")


# --- _make_cv_folds ---


class TestMakeCVFolds:
    def test_loo_one_fold_per_row(self) -> None:
        rows = _oracle_rows()
        folds = _make_cv_folds(rows, "loo")
        assert len(folds) == len(rows)
        for train_idx, test_idx in folds:
            assert len(test_idx) == 1
            assert len(train_idx) == len(rows) - 1
            assert set(train_idx).isdisjoint(test_idx)
            assert sorted(train_idx + test_idx) == list(range(len(rows)))

    def test_noise_holds_out_each_value(self) -> None:
        rows = _oracle_rows()
        folds = _make_cv_folds(rows, "noise")
        # 2 unique noise conditions in our minimal oracle.
        assert len(folds) == 2
        for train_idx, test_idx in folds:
            test_noise_values = {rows[i]["noise_condition"] for i in test_idx}
            train_noise_values = {rows[i]["noise_condition"] for i in train_idx}
            assert len(test_noise_values) == 1
            assert test_noise_values.isdisjoint(train_noise_values)

    def test_delay_holds_out_each_value(self) -> None:
        rows = _oracle_rows()
        folds = _make_cv_folds(rows, "delay")
        assert len(folds) == 2  # 2 unique delays
        for train_idx, test_idx in folds:
            test_delays = {rows[i]["delay_steps"] for i in test_idx}
            train_delays = {rows[i]["delay_steps"] for i in train_idx}
            assert len(test_delays) == 1
            assert test_delays.isdisjoint(train_delays)

    def test_controller_holds_out_each_value(self) -> None:
        rows = _oracle_rows()
        folds = _make_cv_folds(rows, "controller")
        assert len(folds) == 3  # 3 controllers
        for train_idx, test_idx in folds:
            test_ctrls = {rows[i]["controller"] for i in test_idx}
            train_ctrls = {rows[i]["controller"] for i in train_idx}
            assert len(test_ctrls) == 1
            assert test_ctrls.isdisjoint(train_ctrls)

    def test_unknown_strategy(self) -> None:
        with pytest.raises(ValueError, match="unknown cv strategy"):
            _make_cv_folds(_oracle_rows(), "banana")


# --- train_gain_predictor ---


class TestTrainGainPredictor:
    def test_smoke_runs_and_returns_metrics(self) -> None:
        rows = _oracle_rows()
        config = LearnedGainTrainConfig(epochs=50, cv_strategies=("loo",))
        model, metrics = train_gain_predictor(
            rows,
            config=config,
            noise_conditions=_noise_conditions(),
        )
        assert metrics["n_oracle_rows"] == len(rows)
        assert metrics["feature_dim"] == 8
        assert "loo" in metrics["cv_results"]
        assert "final_train_loss_history" in metrics
        assert len(metrics["final_train_loss_history"]) == 50

    def test_final_train_error_decreases(self) -> None:
        # MSE on synthetic oracle should reduce sharply over 200 epochs.
        rows = _oracle_rows()
        config = LearnedGainTrainConfig(epochs=200, cv_strategies=("loo",))
        _model, metrics = train_gain_predictor(
            rows,
            config=config,
            noise_conditions=_noise_conditions(),
        )
        history = metrics["final_train_loss_history"]
        assert history[-1] < history[0]

    def test_cv_results_have_expected_keys(self) -> None:
        rows = _oracle_rows()
        config = LearnedGainTrainConfig(
            epochs=20,
            cv_strategies=("loo", "noise", "delay", "controller"),
        )
        _model, metrics = train_gain_predictor(
            rows,
            config=config,
            noise_conditions=_noise_conditions(),
        )
        for strategy in ("loo", "noise", "delay", "controller"):
            assert strategy in metrics["cv_results"]
            r = metrics["cv_results"][strategy]
            for k in ("n_folds", "abs_error_mean", "abs_error_max", "per_fold"):
                assert k in r

    def test_unknown_noise_condition_in_row_raises(self) -> None:
        rows = _oracle_rows() + [{
            "controller": "random",
            "noise_condition": "extreme",
            "delay_steps": 0,
            "gain": 0.5,
        }]
        config = LearnedGainTrainConfig(epochs=5, cv_strategies=("loo",))
        with pytest.raises(ValueError, match="extreme"):
            train_gain_predictor(
                rows,
                config=config,
                noise_conditions=_noise_conditions(),
            )

    def test_empty_rows_raises(self) -> None:
        config = LearnedGainTrainConfig(epochs=5, cv_strategies=("loo",))
        with pytest.raises(ValueError, match="non-empty"):
            train_gain_predictor(
                [],
                config=config,
                noise_conditions=_noise_conditions(),
            )

    def test_empty_noise_conditions_raises(self) -> None:
        config = LearnedGainTrainConfig(epochs=5, cv_strategies=("loo",))
        with pytest.raises(ValueError, match="non-empty mapping"):
            train_gain_predictor(
                _oracle_rows(),
                config=config,
                noise_conditions={},
            )


def test_seed_reproducibility() -> None:
    rows = _oracle_rows()
    cfg = LearnedGainTrainConfig(epochs=20, cv_strategies=("loo",), seed=42)
    a, _ = train_gain_predictor(rows, config=cfg, noise_conditions=_noise_conditions())
    b, _ = train_gain_predictor(rows, config=cfg, noise_conditions=_noise_conditions())
    for (_, pa), (_, pb) in zip(a.named_parameters(), b.named_parameters()):
        import torch
        torch.testing.assert_close(pa, pb)
