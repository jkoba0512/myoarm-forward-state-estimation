"""Tests for HeuristicReachController (Phase 2 MVP controller)."""

from __future__ import annotations

import numpy as np
import pytest

from myoarm_fse.controllers import (
    Controller,
    HeuristicReachController,
    make_controller,
)
from myoarm_fse.envs.state import MyoArmState, StateSpec


def _state(reach_err: np.ndarray) -> MyoArmState:
    spec = StateSpec(qpos_dim=2, qvel_dim=2, act_dim=3)
    return MyoArmState.from_arrays(
        qpos=np.zeros(spec.qpos_dim),
        qvel=np.zeros(spec.qvel_dim),
        act=np.zeros(spec.act_dim),
        tip_pos=np.zeros(3),
        target_pos=-np.asarray(reach_err, dtype=np.float64),
        # reach_err = tip_pos - target_pos = 0 - (-reach_err) = reach_err
        reach_err=np.asarray(reach_err, dtype=np.float64),
    )


class TestConstruction:
    def test_defaults(self) -> None:
        c = HeuristicReachController(action_dim=34)
        assert c.action_dim == 34
        assert c.gain == 5.0
        assert c.logit_base == 0.0
        assert c.W.shape == (34, 3)
        assert c.W_seed == 0

    def test_invalid_action_dim(self) -> None:
        with pytest.raises(ValueError):
            HeuristicReachController(action_dim=0)
        with pytest.raises(ValueError):
            HeuristicReachController(action_dim=True)  # type: ignore[arg-type]

    def test_invalid_gain(self) -> None:
        with pytest.raises(ValueError):
            HeuristicReachController(action_dim=34, gain=float("nan"))
        with pytest.raises(ValueError):
            HeuristicReachController(action_dim=34, gain=True)  # type: ignore[arg-type]

    def test_invalid_W_seed(self) -> None:
        with pytest.raises(ValueError):
            HeuristicReachController(action_dim=34, W_seed=True)  # type: ignore[arg-type]

    def test_explicit_W(self) -> None:
        W = np.zeros((34, 3), dtype=np.float32)
        c = HeuristicReachController(action_dim=34, W=W)
        np.testing.assert_array_equal(c.W, W)

    def test_explicit_W_bad_shape_raises(self) -> None:
        with pytest.raises(ValueError, match="W must have shape"):
            HeuristicReachController(action_dim=34, W=np.zeros((34, 4)))

    def test_W_seed_deterministic(self) -> None:
        a = HeuristicReachController(action_dim=34, W_seed=123)
        b = HeuristicReachController(action_dim=34, W_seed=123)
        np.testing.assert_array_equal(a.W, b.W)

    def test_W_seed_varies(self) -> None:
        a = HeuristicReachController(action_dim=34, W_seed=1)
        b = HeuristicReachController(action_dim=34, W_seed=2)
        assert not np.array_equal(a.W, b.W)


# --- protocol conformance ---


class TestProtocolConformance:
    def test_controller_protocol(self) -> None:
        c = HeuristicReachController(action_dim=34)
        assert isinstance(c, Controller)

    def test_reset_is_safe_with_or_without_seed(self) -> None:
        c = HeuristicReachController(action_dim=34)
        c.reset()
        c.reset(seed=42)


# --- act() ---


class TestAct:
    def test_action_shape_and_range(self) -> None:
        c = HeuristicReachController(action_dim=34, W_seed=7)
        out = c.act(_state(np.array([0.05, -0.01, 0.02])))
        assert out.shape == (34,)
        assert out.dtype == np.float32
        assert np.all(out >= 0.0) and np.all(out <= 1.0)

    def test_zero_reach_err_yields_sigmoid_of_base(self) -> None:
        c = HeuristicReachController(action_dim=34, logit_base=0.0)
        out = c.act(_state(np.zeros(3)))
        # sigmoid(0) = 0.5 for every muscle.
        np.testing.assert_allclose(out, np.full(34, 0.5, dtype=np.float32), atol=1e-6)

    def test_action_varies_with_reach_err(self) -> None:
        c = HeuristicReachController(action_dim=34, gain=5.0, W_seed=7)
        a = c.act(_state(np.array([0.05, 0.0, 0.0])))
        b = c.act(_state(np.array([-0.05, 0.0, 0.0])))
        # With nontrivial W, equal-magnitude opposite-direction reach
        # errors must produce different actions.
        assert not np.allclose(a, b, atol=1e-4)

    def test_deterministic_same_observation(self) -> None:
        c = HeuristicReachController(action_dim=34, W_seed=7)
        obs = _state(np.array([0.05, -0.02, 0.01]))
        a = c.act(obs)
        b = c.act(obs)
        np.testing.assert_array_equal(a, b)

    def test_logit_base_shifts_baseline(self) -> None:
        # With large negative logit_base, output saturates near 0 at
        # zero reach error; with large positive, near 1.
        lo = HeuristicReachController(action_dim=34, logit_base=-10.0)
        hi = HeuristicReachController(action_dim=34, logit_base=+10.0)
        zero = _state(np.zeros(3))
        np.testing.assert_array_less(lo.act(zero), np.full(34, 0.001))
        np.testing.assert_array_less(np.full(34, 0.999), hi.act(zero))

    def test_non_state_input_raises(self) -> None:
        c = HeuristicReachController(action_dim=34)
        with pytest.raises(ValueError, match="must be MyoArmState"):
            c.act(np.zeros(3))  # type: ignore[arg-type]


# --- factory ---


class TestFactory:
    def test_make_controller_heuristic_reach(self) -> None:
        c = make_controller(
            {"name": "heuristic_reach", "logit_base": 0.0, "gain": 3.0, "W_seed": 11},
            action_dim=34,
            seed=0,
        )
        assert isinstance(c, HeuristicReachController)
        assert c.action_dim == 34
        assert c.gain == 3.0
        assert c.W_seed == 11

    def test_make_controller_default_W_seed_falls_back_to_seed(self) -> None:
        c = make_controller(
            {"name": "heuristic_reach"}, action_dim=34, seed=42,
        )
        assert isinstance(c, HeuristicReachController)
        # When spec omits W_seed the factory must use the caller-supplied seed.
        assert c.W_seed == 42
