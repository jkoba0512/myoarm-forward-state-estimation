"""Tests for JointSpacePDController (Phase 2 D)."""

from __future__ import annotations

import numpy as np
import pytest

from myoarm_fse.controllers import Controller, JointSpacePDController
from myoarm_fse.envs.state import MyoArmState, StateSpec


def _spec(nq: int = 2, na: int = 3) -> StateSpec:
    return StateSpec(qpos_dim=nq, qvel_dim=nq, act_dim=na)


def _state(qpos: np.ndarray, qvel: np.ndarray, *, na: int = 3) -> MyoArmState:
    nq = qpos.shape[0]
    return MyoArmState.from_arrays(
        qpos=qpos.astype(np.float64),
        qvel=qvel.astype(np.float64),
        act=np.zeros(na),
        tip_pos=np.zeros(3),
        target_pos=np.zeros(3),
        reach_err=np.zeros(3),
    )


class TestConstruction:
    def test_defaults(self) -> None:
        c = JointSpacePDController(
            action_dim=3,
            target_qpos=np.array([0.1, -0.2]),
            moment_arm=np.ones((3, 2)),
        )
        assert c.action_dim == 3
        assert c.Kp == 10.0
        assert c.Kd == 1.0
        assert c.action_scale == 0.1
        assert c.target_qpos.shape == (2,)
        assert c.moment_arm.shape == (3, 2)

    def test_invalid_action_dim(self) -> None:
        with pytest.raises(ValueError):
            JointSpacePDController(
                action_dim=0,
                target_qpos=np.zeros(2),
                moment_arm=np.zeros((0, 2)),
            )

    def test_invalid_target_qpos_shape(self) -> None:
        with pytest.raises(ValueError, match="must be 1-D"):
            JointSpacePDController(
                action_dim=3,
                target_qpos=np.zeros((2, 2)),
                moment_arm=np.ones((3, 2)),
            )

    def test_moment_arm_shape_mismatch(self) -> None:
        with pytest.raises(ValueError, match="moment_arm must have shape"):
            JointSpacePDController(
                action_dim=3,
                target_qpos=np.zeros(2),
                moment_arm=np.ones((3, 5)),
            )

    def test_invalid_Kp(self) -> None:
        with pytest.raises(ValueError):
            JointSpacePDController(
                action_dim=3, target_qpos=np.zeros(2),
                moment_arm=np.ones((3, 2)), Kp=float("nan"),
            )


# --- protocol ---


class TestProtocolConformance:
    def test_controller_protocol(self) -> None:
        c = JointSpacePDController(
            action_dim=3, target_qpos=np.zeros(2),
            moment_arm=np.ones((3, 2)),
        )
        assert isinstance(c, Controller)

    def test_reset_accepts_seed(self) -> None:
        c = JointSpacePDController(
            action_dim=3, target_qpos=np.zeros(2),
            moment_arm=np.ones((3, 2)),
        )
        c.reset()
        c.reset(seed=42)


# --- act() ---


class TestAct:
    def test_at_target_with_zero_vel_yields_zero_action(self) -> None:
        # u_joint = Kp * 0 - Kd * 0 = 0 -> drive = 0 -> u_muscle = 0.
        target = np.array([0.1, -0.3], dtype=np.float32)
        c = JointSpacePDController(
            action_dim=3, target_qpos=target,
            moment_arm=np.ones((3, 2)),
        )
        out = c.act(_state(target, np.zeros(2)))
        np.testing.assert_array_equal(out, np.zeros(3, dtype=np.float32))

    def test_action_range(self) -> None:
        c = JointSpacePDController(
            action_dim=3, target_qpos=np.zeros(2),
            moment_arm=np.ones((3, 2)),
        )
        # Far-from-target -> large drive -> clipped to 1.
        out = c.act(_state(qpos=np.array([-10.0, 10.0]), qvel=np.zeros(2)))
        assert np.all(out >= 0.0) and np.all(out <= 1.0)

    def test_drive_sign_decides_activation(self) -> None:
        # moment_arm row 0 positive, row 1 negative on dof 0.
        M = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 0.0]], dtype=np.float32)
        c = JointSpacePDController(
            action_dim=3, target_qpos=np.array([1.0, 0.0]),
            moment_arm=M, Kp=1.0, Kd=0.0, action_scale=1.0,
        )
        # qpos below target on dof 0 -> u_joint[0] > 0
        # drive = M @ u_joint -> [+, -, 0] -> relu -> [+, 0, 0]
        out = c.act(_state(qpos=np.array([0.0, 0.0]), qvel=np.zeros(2)))
        assert out[0] > 0.0
        assert out[1] == 0.0  # antagonist suppressed by ReLU
        assert out[2] == 0.0

    def test_velocity_term_damps_action(self) -> None:
        # If qvel is in target direction, u_joint shrinks.
        M = np.ones((3, 2), dtype=np.float32)
        no_vel = JointSpacePDController(
            action_dim=3, target_qpos=np.array([1.0, 0.0]),
            moment_arm=M, Kp=1.0, Kd=1.0, action_scale=1.0,
        ).act(_state(qpos=np.array([0.0, 0.0]), qvel=np.zeros(2)))
        with_vel = JointSpacePDController(
            action_dim=3, target_qpos=np.array([1.0, 0.0]),
            moment_arm=M, Kp=1.0, Kd=1.0, action_scale=1.0,
        ).act(_state(qpos=np.array([0.0, 0.0]),
                     qvel=np.array([0.5, 0.0])))
        assert with_vel[0] < no_vel[0]

    def test_action_scale_scales(self) -> None:
        M = np.ones((3, 2), dtype=np.float32)
        small = JointSpacePDController(
            action_dim=3, target_qpos=np.array([0.01, 0.0]),
            moment_arm=M, Kp=1.0, Kd=0.0, action_scale=0.5,
        ).act(_state(qpos=np.zeros(2), qvel=np.zeros(2)))
        big = JointSpacePDController(
            action_dim=3, target_qpos=np.array([0.01, 0.0]),
            moment_arm=M, Kp=1.0, Kd=0.0, action_scale=2.0,
        ).act(_state(qpos=np.zeros(2), qvel=np.zeros(2)))
        # Both below saturation; 4x scale -> 4x action.
        np.testing.assert_allclose(big, 4.0 * small, rtol=1e-5)

    def test_state_shape_mismatch_raises(self) -> None:
        c = JointSpacePDController(
            action_dim=3, target_qpos=np.zeros(2),
            moment_arm=np.ones((3, 2)),
        )
        # State has qpos_dim=3, but controller expects 2.
        bad = _state(qpos=np.zeros(3), qvel=np.zeros(3))
        with pytest.raises(ValueError, match="qpos/qvel must have shape"):
            c.act(bad)

    def test_non_state_input_raises(self) -> None:
        c = JointSpacePDController(
            action_dim=3, target_qpos=np.zeros(2),
            moment_arm=np.ones((3, 2)),
        )
        with pytest.raises(ValueError, match="must be MyoArmState"):
            c.act(np.zeros(3))  # type: ignore[arg-type]
