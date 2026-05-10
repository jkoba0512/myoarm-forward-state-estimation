"""Tests for myoarm_fse.metrics.prediction (pure ndarray, no MyoSuite)."""

from __future__ import annotations

import numpy as np
import pytest

from myoarm_fse.metrics import (
    one_step_prediction_mse,
    rollout_mse,
    tip_prediction_error,
)


# --- one_step_prediction_mse ---


class TestOneStepPredictionMSE:
    def test_zero_error(self) -> None:
        a = np.zeros((10, 5), dtype=np.float32)
        assert one_step_prediction_mse(a, a) == 0.0

    def test_known_error(self) -> None:
        # 4 elements, all differ by 0.5 → MSE = 0.25
        true = np.zeros((2, 2), dtype=np.float32)
        pred = np.full((2, 2), 0.5, dtype=np.float32)
        assert one_step_prediction_mse(true, pred) == pytest.approx(0.25)

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="shape"):
            one_step_prediction_mse(
                np.zeros((10, 5)), np.zeros((10, 4))
            )

    def test_ndim_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="2-D"):
            one_step_prediction_mse(np.zeros(10), np.zeros(10))

    def test_nan_in_true_raises(self) -> None:
        a = np.zeros((3, 3))
        a[0, 0] = np.nan
        with pytest.raises(ValueError, match="non-finite"):
            one_step_prediction_mse(a, np.zeros((3, 3)))

    def test_inf_in_pred_raises(self) -> None:
        a = np.zeros((3, 3))
        a[0, 0] = np.inf
        with pytest.raises(ValueError, match="non-finite"):
            one_step_prediction_mse(np.zeros((3, 3)), a)

    def test_empty_returns_zero(self) -> None:
        a = np.zeros((0, 5), dtype=np.float32)
        assert one_step_prediction_mse(a, a) == 0.0

    def test_dtype_independent(self) -> None:
        # float64 inputs should also work.
        a = np.zeros((3, 3), dtype=np.float64)
        b = np.full((3, 3), 0.1, dtype=np.float64)
        assert one_step_prediction_mse(a, b) == pytest.approx(0.01)


# --- rollout_mse ---


class TestRolloutMSE:
    def test_zero_error(self) -> None:
        a = np.zeros((100, 10), dtype=np.float32)
        assert rollout_mse(a, a) == 0.0

    def test_known_error(self) -> None:
        true = np.zeros((4, 3), dtype=np.float32)
        pred = np.ones((4, 3), dtype=np.float32)
        assert rollout_mse(true, pred) == pytest.approx(1.0)

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="shape"):
            rollout_mse(np.zeros((50, 10)), np.zeros((50, 5)))

    def test_ndim_must_be_2(self) -> None:
        with pytest.raises(ValueError, match="2-D"):
            rollout_mse(np.zeros(50), np.zeros(50))

    def test_empty_returns_zero(self) -> None:
        assert rollout_mse(np.zeros((0, 5)), np.zeros((0, 5))) == 0.0


# --- tip_prediction_error ---


class TestTipPredictionError:
    def test_zero_error(self) -> None:
        a = np.zeros((10, 3), dtype=np.float32)
        assert tip_prediction_error(a, a) == 0.0

    def test_unit_error(self) -> None:
        # ||[1, 0, 0]|| = 1, mean over 5 steps = 1
        true = np.zeros((5, 3), dtype=np.float32)
        pred = np.tile([1.0, 0.0, 0.0], (5, 1)).astype(np.float32)
        assert tip_prediction_error(true, pred) == pytest.approx(1.0)

    def test_3_4_5_norm(self) -> None:
        # All steps differ by [3, 4, 0] → mean distance = 5
        true = np.zeros((4, 3), dtype=np.float32)
        pred = np.tile([3.0, 4.0, 0.0], (4, 1)).astype(np.float32)
        assert tip_prediction_error(true, pred) == pytest.approx(5.0)

    def test_shape_must_be_T_3(self) -> None:
        with pytest.raises(ValueError, match=r"\(T, 3\)"):
            tip_prediction_error(np.zeros((5, 4)), np.zeros((5, 4)))

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="shape"):
            tip_prediction_error(np.zeros((5, 3)), np.zeros((6, 3)))

    def test_nan_raises(self) -> None:
        a = np.zeros((3, 3))
        a[0, 0] = np.nan
        with pytest.raises(ValueError, match="non-finite"):
            tip_prediction_error(a, np.zeros((3, 3)))

    def test_empty_returns_zero(self) -> None:
        a = np.zeros((0, 3), dtype=np.float32)
        assert tip_prediction_error(a, a) == 0.0
