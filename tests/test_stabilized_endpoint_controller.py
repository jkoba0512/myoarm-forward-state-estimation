"""Tests for StabilizedEndpointController (Stage B controller pivot).

Codex required five minimum-test coverage points (per the response
``2026-05-27_stage-b-controller-pivot-proposal_response_to-claude-code``):

  (i)   controller dispatch can invoke ``name=stabilized_endpoint``
  (ii)  1-episode smoke test completes without NaN
  (iii) at K=1 oracle, noise=none, delay=0, tip movement is meaningfully
        larger than the joint-PD baseline
  (iv)  scipy / nnls unavailable → explicit ImportError
  (v)   T_ramp=0 / None is explicitly handled

(i)-(iii) are env-integration tests and live in
``test_stabilized_endpoint_smoke.py``. This file holds the unit tests
((iv), (v), construction validation, protocol conformance).
"""

from __future__ import annotations

import importlib
import sys

import numpy as np
import pytest

from myoarm_fse.controllers import (
    Controller,
    StabilizedEndpointController,
)
from myoarm_fse.envs.state import MyoArmState, StateSpec


def _state(qvel: np.ndarray) -> MyoArmState:
    """Build a MyoArmState with the given qvel and zero tip/target."""
    nv = qvel.shape[0]
    return MyoArmState.from_arrays(
        qpos=np.zeros(nv, dtype=np.float64),
        qvel=qvel.astype(np.float64),
        act=np.zeros(3, dtype=np.float64),
        tip_pos=np.zeros(3, dtype=np.float64),
        target_pos=np.zeros(3, dtype=np.float64),
        reach_err=np.zeros(3, dtype=np.float64),
    )


def _build(*, nv: int = 4, nu: int = 6, T_ramp=300,
           record_history: bool = False) -> StabilizedEndpointController:
    rng = np.random.default_rng(0)
    return StabilizedEndpointController(
        action_dim=nu,
        init_tip=np.array([0.0, 0.0, 0.5], dtype=np.float32),
        target_pos=np.array([0.1, -0.1, 1.0], dtype=np.float32),
        jacobian=rng.normal(size=(3, nv)).astype(np.float32),
        moment_arm=rng.normal(size=(nu, nv)).astype(np.float32),
        Kp=30.0, Kd=3.0, action_scale=5.0,
        T_ramp=T_ramp,
        record_history=record_history,
    )


# --- construction --------------------------------------------------------


class TestConstruction:
    def test_defaults(self) -> None:
        c = _build()
        assert c.action_dim == 6
        assert c.nv == 4
        assert c.Kp == 30.0 and c.Kd == 3.0 and c.action_scale == 5.0
        assert c.T_ramp == 300
        assert c.init_tip.shape == (3,)
        assert c.target_pos.shape == (3,)
        assert c.jacobian.shape == (3, 4)
        assert c.moment_arm.shape == (6, 4)

    @pytest.mark.parametrize("bad", [0, -1, True, 3.0])
    def test_invalid_action_dim(self, bad) -> None:
        with pytest.raises(ValueError):
            StabilizedEndpointController(
                action_dim=bad,
                init_tip=np.zeros(3), target_pos=np.zeros(3),
                jacobian=np.zeros((3, 4)), moment_arm=np.zeros((6, 4)),
            )

    def test_invalid_init_tip_shape(self) -> None:
        with pytest.raises(ValueError, match="init_tip must have shape"):
            StabilizedEndpointController(
                action_dim=6, init_tip=np.zeros(4),
                target_pos=np.zeros(3),
                jacobian=np.zeros((3, 4)), moment_arm=np.zeros((6, 4)),
            )

    def test_init_tip_non_finite(self) -> None:
        bad = np.array([0.0, np.nan, 0.0])
        with pytest.raises(ValueError, match="non-finite"):
            StabilizedEndpointController(
                action_dim=6, init_tip=bad, target_pos=np.zeros(3),
                jacobian=np.zeros((3, 4)), moment_arm=np.zeros((6, 4)),
            )

    def test_target_pos_non_finite(self) -> None:
        bad = np.array([0.0, 0.0, np.inf])
        with pytest.raises(ValueError, match="non-finite"):
            StabilizedEndpointController(
                action_dim=6, init_tip=np.zeros(3), target_pos=bad,
                jacobian=np.zeros((3, 4)), moment_arm=np.zeros((6, 4)),
            )

    def test_jacobian_shape_mismatch(self) -> None:
        with pytest.raises(ValueError, match="jacobian must have shape"):
            StabilizedEndpointController(
                action_dim=6, init_tip=np.zeros(3),
                target_pos=np.zeros(3),
                jacobian=np.zeros((2, 4)),  # 2 instead of 3
                moment_arm=np.zeros((6, 4)),
            )

    def test_moment_arm_shape_mismatch(self) -> None:
        with pytest.raises(ValueError, match="moment_arm must have shape"):
            StabilizedEndpointController(
                action_dim=6, init_tip=np.zeros(3),
                target_pos=np.zeros(3),
                jacobian=np.zeros((3, 4)),
                moment_arm=np.zeros((6, 5)),  # 5 cols, jac says nv=4
            )

    @pytest.mark.parametrize("Kp", [float("nan"), float("inf"), True])
    def test_invalid_Kp(self, Kp) -> None:
        with pytest.raises(ValueError):
            StabilizedEndpointController(
                action_dim=6, init_tip=np.zeros(3),
                target_pos=np.zeros(3),
                jacobian=np.zeros((3, 4)),
                moment_arm=np.zeros((6, 4)),
                Kp=Kp,
            )


# --- T_ramp behaviour (Codex test point (v)) -----------------------------


class TestTRamp:
    def test_T_ramp_none_disables(self) -> None:
        c = _build(T_ramp=None)
        # After reset, step 0 should already see s=1 (final target).
        assert c.T_ramp == 0
        _ = c.act(_state(np.zeros(4)))
        assert c.last_ramp_progress == 1.0

    def test_T_ramp_zero_disables(self) -> None:
        c = _build(T_ramp=0)
        assert c.T_ramp == 0
        _ = c.act(_state(np.zeros(4)))
        assert c.last_ramp_progress == 1.0

    def test_T_ramp_negative_rejected(self) -> None:
        with pytest.raises(ValueError, match=r"T_ramp must be >= 0"):
            _build(T_ramp=-1)

    def test_T_ramp_bool_rejected(self) -> None:
        with pytest.raises(ValueError, match="T_ramp must be"):
            _build(T_ramp=True)

    def test_T_ramp_float_rejected(self) -> None:
        with pytest.raises(ValueError, match="T_ramp must be"):
            _build(T_ramp=300.0)

    def test_ramp_progresses_linearly(self) -> None:
        # ``act()`` computes s = step / T_ramp using the step counter
        # BEFORE its post-increment; so after N calls the last s is
        # (N - 1) / T_ramp (clamped at 1).
        c = _build(T_ramp=100)
        c.act(_state(np.zeros(4)))  # 1 call → s = 0/100 = 0
        assert c.last_ramp_progress == pytest.approx(0.0)
        for _ in range(50):           # 51 calls → s = 50/100 = 0.5
            c.act(_state(np.zeros(4)))
        assert c.last_ramp_progress == pytest.approx(0.5)
        for _ in range(50):           # 101 calls → s = 100/100 = 1.0
            c.act(_state(np.zeros(4)))
        assert c.last_ramp_progress == pytest.approx(1.0)
        # further calls remain capped
        c.act(_state(np.zeros(4)))
        assert c.last_ramp_progress == pytest.approx(1.0)


# --- protocol conformance ------------------------------------------------


class TestProtocol:
    def test_protocol(self) -> None:
        c = _build()
        assert isinstance(c, Controller)

    def test_act_output_shape_dtype_range(self) -> None:
        c = _build(nv=4, nu=6)
        out = c.act(_state(np.zeros(4)))
        assert isinstance(out, np.ndarray)
        assert out.shape == (6,)
        assert out.dtype == np.float32
        assert np.all(out >= 0.0) and np.all(out <= 1.0)

    def test_act_non_nan_with_nonzero_state(self) -> None:
        c = _build()
        out = c.act(_state(np.array([0.1, -0.2, 0.3, 0.0])))
        assert np.isfinite(out).all()


# --- diagnostics / history ----------------------------------------------


class TestDiagnostics:
    def test_record_history_off_by_default(self) -> None:
        c = _build()
        for _ in range(5):
            c.act(_state(np.zeros(4)))
        assert c.nnls_residual_history == []
        assert c.ramp_progress_history == []
        assert c.activation_history == []
        # last_* still populated
        assert c.last_ramp_progress >= 0.0

    def test_record_history_on(self) -> None:
        c = _build(record_history=True)
        for _ in range(5):
            c.act(_state(np.zeros(4)))
        assert len(c.nnls_residual_history) == 5
        assert len(c.ramp_progress_history) == 5
        assert len(c.activation_history) == 5
        assert all(r >= 0.0 for r in c.nnls_residual_history)
        assert all(0.0 <= r <= 1.0 for r in c.ramp_progress_history)
        for a in c.activation_history:
            assert a.shape == (6,)
            assert np.all((a >= 0.0) & (a <= 1.0))

    def test_reset_clears_history_and_step(self) -> None:
        c = _build(record_history=True)
        for _ in range(5):
            c.act(_state(np.zeros(4)))
        assert len(c.nnls_residual_history) == 5
        c.reset()
        assert c.nnls_residual_history == []
        assert c.ramp_progress_history == []
        assert c.activation_history == []
        # step counter reset → first act after reset should see s=0
        # (assuming T_ramp > 0)
        c.act(_state(np.zeros(4)))
        assert c.last_ramp_progress == pytest.approx(0.0)


# --- scipy / nnls import failure (Codex test point (iv)) ---------------


class TestScipyImport:
    """Confirm StabilizedEndpointController fails clearly without scipy.

    The controller imports ``scipy.optimize.nnls`` at module import time
    and wraps an ImportError. We simulate scipy absence by reloading the
    module with ``scipy.optimize`` poisoned in ``sys.modules``.
    """

    def test_module_import_fails_without_scipy(self) -> None:
        target_mod = "myoarm_fse.controllers.stabilized_endpoint"
        scipy_mod = "scipy.optimize"
        # Save the originals and arrange to restore them.
        saved_target = sys.modules.pop(target_mod, None)
        saved_scipy = sys.modules.pop(scipy_mod, None)
        sys.modules[scipy_mod] = None  # type: ignore[assignment]
        try:
            with pytest.raises(ImportError, match="scipy"):
                importlib.import_module(target_mod)
        finally:
            if saved_scipy is not None:
                sys.modules[scipy_mod] = saved_scipy
            else:
                sys.modules.pop(scipy_mod, None)
            if saved_target is not None:
                sys.modules[target_mod] = saved_target
            else:
                sys.modules.pop(target_mod, None)


# --- __repr__ smoke ------------------------------------------------------


def test_repr() -> None:
    c = _build()
    r = repr(c)
    assert "StabilizedEndpointController" in r
    assert "action_dim=6" in r
    assert "nv=4" in r
    assert "T_ramp=300" in r
