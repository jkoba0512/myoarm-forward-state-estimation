"""Endpoint-error feedback controller with Jacobian-transpose mapping.

This is a task-level analogue of the joint-space PD controller in
``joint_pd.py``. Instead of pre-solving inverse kinematics, the
controller drives the tip toward the target by a proportional-
derivative law in Cartesian (tip) space and then projects the
Cartesian command into muscle-activation space via Jacobian transpose
and the moment-arm matrix.

Per-step action::

    e_tip   = target_pos - tip_pos_est                # (3,)
    v_tip   = jacobian @ qvel_est                     # (3,)
    u_tip   = Kp_e * e_tip - Kd_e * v_tip             # (3,)
    u_joint = jacobian.T @ u_tip                      # (nv,)
    drive   = moment_arm @ u_joint                    # (nu,)
    u_muscle = clip(action_scale * relu(drive), 0, 1) # (nu,)

This is the classic Jacobian-transpose method for task-space
manipulation; it is well-known not to be optimal (no Riccati / LQR /
OFC is solved here). The controller is meant as a transparent,
state-coupled, task-level feedback law that lets the observer output
flow directly to muscle commands without an IK pre-step --- a third
class of controller beside the joint-space PD feedback law and the
behaviour-cloned policy.
"""

from __future__ import annotations

import numpy as np

from myoarm_fse.envs.state import MyoArmState

_DTYPE: np.dtype = np.dtype(np.float32)
_LO: float = 0.0
_HI: float = 1.0
_CART_DIM: int = 3


class EndpointErrorFeedbackController:
    """Task-level PD on tip-to-target error, projected through J^T and M.

    Like ``JointSpacePDController``, the Jacobian and moment-arm matrix
    are captured once at episode start (at the env's neutral pose). This
    is a controller-design choice, not an optimal-control claim.
    """

    def __init__(
        self,
        action_dim: int,
        target_pos: np.ndarray,
        jacobian: np.ndarray,
        moment_arm: np.ndarray,
        *,
        Kp: float = 30.0,
        Kd: float = 3.0,
        action_scale: float = 5.0,
    ) -> None:
        if isinstance(action_dim, bool):
            raise ValueError(
                f"action_dim must be a positive int, got bool: {action_dim!r}"
            )
        if not isinstance(action_dim, int):
            raise ValueError(
                f"action_dim must be a positive int, "
                f"got {type(action_dim).__name__}: {action_dim!r}"
            )
        if action_dim <= 0:
            raise ValueError(f"action_dim must be > 0, got {action_dim}")

        target_arr = np.asarray(target_pos, dtype=_DTYPE)
        if target_arr.shape != (_CART_DIM,):
            raise ValueError(
                f"target_pos must have shape ({_CART_DIM},), "
                f"got {target_arr.shape}"
            )
        if not np.isfinite(target_arr).all():
            raise ValueError("target_pos contains non-finite values")

        jac_arr = np.asarray(jacobian, dtype=_DTYPE)
        if jac_arr.ndim != 2 or jac_arr.shape[0] != _CART_DIM:
            raise ValueError(
                f"jacobian must have shape ({_CART_DIM}, nv), "
                f"got {jac_arr.shape}"
            )
        nv = int(jac_arr.shape[1])

        moment_arr = np.asarray(moment_arm, dtype=_DTYPE)
        if moment_arr.shape != (action_dim, nv):
            raise ValueError(
                f"moment_arm must have shape ({action_dim}, {nv}), "
                f"got {moment_arr.shape}"
            )

        for name, val in (("Kp", Kp), ("Kd", Kd), ("action_scale", action_scale)):
            if isinstance(val, bool):
                raise ValueError(f"{name} must not be bool, got {val!r}")
            if not isinstance(val, (int, float, np.floating)):
                raise ValueError(
                    f"{name} must be numeric, got {type(val).__name__}"
                )
            if not np.isfinite(float(val)):
                raise ValueError(f"{name} must be finite, got {val!r}")

        self._action_dim: int = int(action_dim)
        self._target_pos: np.ndarray = target_arr.copy()
        self._jacobian: np.ndarray = jac_arr.copy()
        self._moment_arm: np.ndarray = moment_arr.copy()
        self._Kp: float = float(Kp)
        self._Kd: float = float(Kd)
        self._action_scale: float = float(action_scale)
        self._nv: int = nv

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def target_pos(self) -> np.ndarray:
        return self._target_pos.copy()

    @property
    def jacobian(self) -> np.ndarray:
        return self._jacobian.copy()

    @property
    def moment_arm(self) -> np.ndarray:
        return self._moment_arm.copy()

    @property
    def Kp(self) -> float:
        return self._Kp

    @property
    def Kd(self) -> float:
        return self._Kd

    @property
    def action_scale(self) -> float:
        return self._action_scale

    def reset(self, *, seed: int | None = None) -> None:
        del seed

    def act(self, observation: MyoArmState) -> np.ndarray:
        if not isinstance(observation, MyoArmState):
            raise ValueError(
                f"observation must be MyoArmState, got {type(observation).__name__}"
            )
        tip_est = np.asarray(observation.tip_pos, dtype=_DTYPE)
        qvel = np.asarray(observation.qvel, dtype=_DTYPE)
        if tip_est.shape != (_CART_DIM,):
            raise ValueError(
                f"observation tip_pos must have shape ({_CART_DIM},), "
                f"got {tip_est.shape}"
            )
        if qvel.shape != (self._nv,):
            raise ValueError(
                f"observation qvel must have shape ({self._nv},), "
                f"got {qvel.shape}"
            )

        # Cartesian (task-space) PD on tip
        e_tip = self._target_pos - tip_est                       # (3,)
        v_tip = self._jacobian @ qvel                            # (3,)
        u_tip = self._Kp * e_tip - self._Kd * v_tip              # (3,)

        # Jacobian-transpose to joint space, then moment-arm to muscle
        u_joint = self._jacobian.T @ u_tip                       # (nv,)
        drive = self._moment_arm @ u_joint                       # (nu,)
        u_muscle = np.clip(
            self._action_scale * np.maximum(drive, 0.0), _LO, _HI,
        )
        return u_muscle.astype(_DTYPE, copy=False)

    def __repr__(self) -> str:
        return (
            f"EndpointErrorFeedbackController(action_dim={self._action_dim}, "
            f"nv={self._nv}, Kp={self._Kp}, Kd={self._Kd}, "
            f"action_scale={self._action_scale})"
        )
