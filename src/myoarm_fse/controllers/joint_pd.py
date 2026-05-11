"""Joint-space PD reach controller with moment-arm muscle mapping (D).

Episode-time setup:

1. Pre-solve IK on the env: ``target_pos`` -> ``target_qpos``
   (``envs.ik.solve_ik``).
2. Capture the env's moment-arm matrix at the neutral pose
   (``envs.ik.actuator_moment_dense``), shape ``(nu, nv)``.
3. Construct ``JointSpacePDController(target_qpos, Kp, Kd, moment_arm,
   action_scale, ...)`` and inject it into ``run_closed_loop_episode``.

Per-step action::

    u_joint  = Kp * (target_qpos - qpos_est) - Kd * qvel_est        # (nv,)
    drive    = moment_arm @ u_joint                                  # (nu,)
    u_muscle = clip(action_scale * relu(drive), 0, 1)                # (nu,)

The ``relu(drive)`` step uses each muscle only in its agonist
direction; antagonist muscles stay quiescent. ``action_scale`` keeps
the overall activation in a reasonable range without per-axis tuning.
"""

from __future__ import annotations

import numpy as np

from myoarm_fse.envs.state import MyoArmState

_DTYPE: np.dtype = np.dtype(np.float32)
_LO: float = 0.0
_HI: float = 1.0


class JointSpacePDController:
    """PD controller in joint space + moment-arm muscle projection."""

    def __init__(
        self,
        action_dim: int,
        target_qpos: np.ndarray,
        moment_arm: np.ndarray,
        *,
        Kp: float = 10.0,
        Kd: float = 1.0,
        action_scale: float = 0.1,
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

        target_arr = np.asarray(target_qpos, dtype=_DTYPE)
        if target_arr.ndim != 1:
            raise ValueError(
                f"target_qpos must be 1-D, got shape {target_arr.shape}"
            )
        if not np.isfinite(target_arr).all():
            raise ValueError("target_qpos contains non-finite values")

        moment_arr = np.asarray(moment_arm, dtype=_DTYPE)
        if moment_arr.shape != (action_dim, target_arr.shape[0]):
            raise ValueError(
                f"moment_arm must have shape ({action_dim}, {target_arr.shape[0]}), "
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
        self._target_qpos: np.ndarray = target_arr.copy()
        self._moment_arm: np.ndarray = moment_arr.copy()
        self._Kp: float = float(Kp)
        self._Kd: float = float(Kd)
        self._action_scale: float = float(action_scale)
        self._nv: int = int(target_arr.shape[0])

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def target_qpos(self) -> np.ndarray:
        return self._target_qpos.copy()

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
        qpos = np.asarray(observation.qpos, dtype=_DTYPE)
        qvel = np.asarray(observation.qvel, dtype=_DTYPE)
        if qpos.shape != (self._nv,) or qvel.shape != (self._nv,):
            raise ValueError(
                f"observation qpos/qvel must have shape ({self._nv},), "
                f"got qpos={qpos.shape}, qvel={qvel.shape}"
            )
        u_joint = self._Kp * (self._target_qpos - qpos) - self._Kd * qvel
        drive = self._moment_arm @ u_joint
        u_muscle = np.clip(
            self._action_scale * np.maximum(drive, 0.0), _LO, _HI,
        )
        return u_muscle.astype(_DTYPE, copy=False)

    def __repr__(self) -> str:
        return (
            f"JointSpacePDController(action_dim={self._action_dim}, "
            f"nv={self._nv}, Kp={self._Kp}, Kd={self._Kd}, "
            f"action_scale={self._action_scale})"
        )
