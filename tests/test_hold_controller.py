"""Tests for myoarm_fse.controllers.hold.HoldController."""

from __future__ import annotations

import numpy as np
import pytest

from myoarm_fse.controllers import Controller, HoldController
from myoarm_fse.envs.state import MyoArmState


def _obs(act_dim: int = 4) -> MyoArmState:
    return MyoArmState.from_arrays(
        qpos=np.zeros(2, dtype=np.float32),
        qvel=np.zeros(3, dtype=np.float32),
        act=np.zeros(act_dim, dtype=np.float32),
        tip_pos=np.zeros(3, dtype=np.float32),
        target_pos=np.zeros(3, dtype=np.float32),
        reach_err=np.zeros(3, dtype=np.float32),
    )


# --- constructor validation ---


class TestConstructor:
    def test_valid(self) -> None:
        c = HoldController(action_dim=4, value=0.3)
        assert c.action_dim == 4
        assert c.value == pytest.approx(0.3)

    def test_default_value_is_zero(self) -> None:
        c = HoldController(action_dim=4)
        assert c.value == 0.0

    @pytest.mark.parametrize("bad", [0, -1])
    def test_non_positive_action_dim(self, bad: int) -> None:
        with pytest.raises(ValueError):
            HoldController(action_dim=bad)

    def test_bool_action_dim(self) -> None:
        with pytest.raises(ValueError):
            HoldController(action_dim=True)  # type: ignore[arg-type]

    def test_float_action_dim(self) -> None:
        with pytest.raises(ValueError):
            HoldController(action_dim=4.0)  # type: ignore[arg-type]

    def test_bool_value(self) -> None:
        with pytest.raises(ValueError):
            HoldController(action_dim=4, value=True)  # type: ignore[arg-type]

    def test_str_value(self) -> None:
        with pytest.raises(ValueError):
            HoldController(action_dim=4, value="0.5")  # type: ignore[arg-type]

    def test_nan_value(self) -> None:
        with pytest.raises(ValueError):
            HoldController(action_dim=4, value=float("nan"))

    def test_inf_value(self) -> None:
        with pytest.raises(ValueError):
            HoldController(action_dim=4, value=float("inf"))

    def test_value_below_zero(self) -> None:
        with pytest.raises(ValueError, match=r"\[0.0, 1.0\]"):
            HoldController(action_dim=4, value=-0.1)

    def test_value_above_one(self) -> None:
        with pytest.raises(ValueError, match=r"\[0.0, 1.0\]"):
            HoldController(action_dim=4, value=1.5)

    def test_value_at_boundaries(self) -> None:
        HoldController(action_dim=4, value=0.0)
        HoldController(action_dim=4, value=1.0)


# --- output ---


class TestActOutput:
    def test_shape_and_dtype(self) -> None:
        c = HoldController(action_dim=34, value=0.5)
        out = c.act(_obs(34))
        assert out.shape == (34,)
        assert out.dtype == np.float32

    def test_values_match_value(self) -> None:
        c = HoldController(action_dim=4, value=0.3)
        out = c.act(_obs(4))
        np.testing.assert_array_equal(out, np.full(4, 0.3, dtype=np.float32))

    def test_zero_default(self) -> None:
        c = HoldController(action_dim=4)
        out = c.act(_obs(4))
        np.testing.assert_array_equal(out, np.zeros(4, dtype=np.float32))

    def test_observation_is_ignored(self) -> None:
        c = HoldController(action_dim=4, value=0.5)
        # Different observations → identical output.
        obs_a = _obs(4)
        obs_b = MyoArmState.from_arrays(
            qpos=np.full(2, 99.0, dtype=np.float32),
            qvel=np.full(3, 99.0, dtype=np.float32),
            act=np.full(4, 99.0, dtype=np.float32),
            tip_pos=np.full(3, 99.0, dtype=np.float32),
            target_pos=np.full(3, 99.0, dtype=np.float32),
            reach_err=np.full(3, 99.0, dtype=np.float32),
        )
        np.testing.assert_array_equal(c.act(obs_a), c.act(obs_b))

    def test_returns_independent_arrays(self) -> None:
        c = HoldController(action_dim=4, value=0.5)
        out1 = c.act(_obs(4))
        out1[0] = 999.0
        out2 = c.act(_obs(4))
        # Mutation of a returned array must not leak into subsequent calls.
        assert out2[0] == 0.5

    def test_act_rejects_non_state(self) -> None:
        c = HoldController(action_dim=4)
        with pytest.raises(ValueError):
            c.act([0, 0, 0])  # type: ignore[arg-type]


# --- reset ---


class TestReset:
    def test_reset_is_no_op(self) -> None:
        c = HoldController(action_dim=4, value=0.7)
        a = c.act(_obs(4))
        c.reset()
        b = c.act(_obs(4))
        np.testing.assert_array_equal(a, b)

    def test_reset_with_seed_is_no_op(self) -> None:
        c = HoldController(action_dim=4, value=0.7)
        c.reset(seed=42)
        np.testing.assert_array_equal(
            c.act(_obs(4)), np.full(4, 0.7, dtype=np.float32)
        )

    def test_reset_bool_seed_raises(self) -> None:
        c = HoldController(action_dim=4)
        with pytest.raises(TypeError):
            c.reset(seed=True)  # type: ignore[arg-type]

    def test_reset_str_seed_raises(self) -> None:
        c = HoldController(action_dim=4)
        with pytest.raises(TypeError):
            c.reset(seed="42")  # type: ignore[arg-type]


# --- protocol conformance ---


def test_implements_controller_protocol() -> None:
    c = HoldController(action_dim=4)
    assert isinstance(c, Controller)
