"""Stabilized endpoint feedback controller with NNLS muscle routing
and virtual target ramping.

Replaces the ``ReLU + clip`` muscle drive of
``EndpointErrorFeedbackController`` with a non-negative least squares
projection that respects the non-negative muscle activation constraint
without symmetry-breaking through ReLU, and ramps the controller's
effective endpoint target from the initial tip toward the final target
over ``T_ramp`` steps to avoid the impulsive endpoint command at
``t = 0`` that destabilises the un-ramped controller.

Per-step action::

    s         = min(step / T_ramp, 1.0)             # ramp progress
    cur_tgt   = init_tip + s * (final_tgt - init_tip)
    e_tip     = cur_tgt - tip_pos_est                (3,)
    v_tip     = J @ qvel_est                         (3,)
    u_tip     = Kp * e_tip - Kd * v_tip              (3,)
    u_joint   = J^T @ u_tip                          (nv,)
    a, r      = nnls(M^T, u_joint)                   # a in R+^(nu,)
    u_muscle  = clip(action_scale * a, 0, 1)         (nu,)

``J`` (tip Jacobian) and ``M`` (actuator moment-arm matrix) are
captured once at episode start at the env's neutral pose; this is a
deliberate design choice for parity with the other state-coupled
controllers in this module.

This controller is a **stabilized probe controller** used as a
post-pivot Stage B baseline (replacing the joint-PD probe that
exposed an artefactual K=0 optimum). It is **not** a biological motor
planner — see exchange
``2026-05-27_stage-b-controller-pivot-proposal_response_to-claude-code``.
"""

from __future__ import annotations

import numpy as np

try:
    from scipy.optimize import nnls as _scipy_nnls
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "StabilizedEndpointController requires scipy "
        "(scipy.optimize.nnls). Install scipy to use this controller."
    ) from e

from myoarm_fse.envs.state import MyoArmState

_DTYPE: np.dtype = np.dtype(np.float32)
_LO: float = 0.0
_HI: float = 1.0
_CART_DIM: int = 3


class StabilizedEndpointController:
    """Task-space PD with NNLS muscle routing and virtual target ramping.

    Like ``EndpointErrorFeedbackController`` and
    ``JointSpacePDController``, ``J`` and ``M`` are captured at
    episode start at the env's neutral pose. The ramp is internal to
    the controller; the external API still receives the final target
    only.

    Parameters
    ----------
    action_dim : int
        Number of muscles / actuators (= ``M.shape[0]``).
    init_tip : (3,) array-like
        The tip position at episode reset (used as the ramp start).
    target_pos : (3,) array-like
        The final endpoint target.
    jacobian : (3, nv) array-like
        Tip Jacobian captured at episode start.
    moment_arm : (nu, nv) array-like
        Actuator moment-arm matrix captured at episode start.
    Kp, Kd : float
        Task-space PD gains.
    action_scale : float
        Multiplier on the NNLS solution before ``[0, 1]`` clipping.
    T_ramp : int or None, default 300
        Linear interpolation length from ``init_tip`` to ``target_pos``.
        ``0`` or ``None`` disables the ramp (controller behaves like a
        static-target task-space PD with NNLS routing).
    record_history : bool, default False
        If ``True``, retain per-step ``nnls_residual_history``,
        ``ramp_progress_history``, and ``activation_history`` lists.
        The most recent step's values are always available via the
        ``last_*`` properties regardless of this flag.
    """

    def __init__(
        self,
        action_dim: int,
        init_tip: np.ndarray,
        target_pos: np.ndarray,
        jacobian: np.ndarray,
        moment_arm: np.ndarray,
        *,
        Kp: float = 30.0,
        Kd: float = 3.0,
        action_scale: float = 5.0,
        T_ramp: int | None = 300,
        record_history: bool = False,
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

        init_arr = np.asarray(init_tip, dtype=_DTYPE)
        if init_arr.shape != (_CART_DIM,):
            raise ValueError(
                f"init_tip must have shape ({_CART_DIM},), "
                f"got {init_arr.shape}"
            )
        if not np.isfinite(init_arr).all():
            raise ValueError("init_tip contains non-finite values")

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

        for name, val in (("Kp", Kp), ("Kd", Kd),
                          ("action_scale", action_scale)):
            if isinstance(val, bool):
                raise ValueError(f"{name} must not be bool, got {val!r}")
            if not isinstance(val, (int, float, np.floating)):
                raise ValueError(
                    f"{name} must be numeric, got {type(val).__name__}"
                )
            if not np.isfinite(float(val)):
                raise ValueError(f"{name} must be finite, got {val!r}")

        # T_ramp: int >= 0 or None; both 0 and None disable the ramp.
        if T_ramp is None:
            T_ramp_eff = 0
        elif isinstance(T_ramp, bool):
            raise ValueError(
                f"T_ramp must be a non-negative int or None, got {T_ramp!r}"
            )
        elif isinstance(T_ramp, int):
            if T_ramp < 0:
                raise ValueError(
                    f"T_ramp must be >= 0 (or None), got {T_ramp}"
                )
            T_ramp_eff = int(T_ramp)
        else:
            raise ValueError(
                f"T_ramp must be a non-negative int or None, "
                f"got {type(T_ramp).__name__}: {T_ramp!r}"
            )

        self._action_dim: int = int(action_dim)
        self._init_tip: np.ndarray = init_arr.copy()
        self._final_target: np.ndarray = target_arr.copy()
        self._jacobian: np.ndarray = jac_arr.copy()
        self._moment_arm: np.ndarray = moment_arr.copy()
        # NNLS solves min ||A x - b||  s.t. x >= 0.
        # We want activation >= 0 s.t. M^T @ a ≈ u_joint, so A = M^T.
        self._mt: np.ndarray = moment_arr.astype(np.float64).T.copy()
        self._Kp: float = float(Kp)
        self._Kd: float = float(Kd)
        self._action_scale: float = float(action_scale)
        self._T_ramp: int = T_ramp_eff
        self._nv: int = nv

        self._step: int = 0
        self._record_history: bool = bool(record_history)
        self._nnls_residual_history: list[float] = []
        self._ramp_progress_history: list[float] = []
        self._activation_history: list[np.ndarray] = []
        self._last_nnls_residual: float = 0.0
        self._last_ramp_progress: float = 0.0

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def nv(self) -> int:
        return self._nv

    @property
    def Kp(self) -> float:
        return self._Kp

    @property
    def Kd(self) -> float:
        return self._Kd

    @property
    def action_scale(self) -> float:
        return self._action_scale

    @property
    def T_ramp(self) -> int:
        return self._T_ramp

    @property
    def init_tip(self) -> np.ndarray:
        return self._init_tip.copy()

    @property
    def target_pos(self) -> np.ndarray:
        return self._final_target.copy()

    @property
    def jacobian(self) -> np.ndarray:
        return self._jacobian.copy()

    @property
    def moment_arm(self) -> np.ndarray:
        return self._moment_arm.copy()

    @property
    def last_nnls_residual(self) -> float:
        return self._last_nnls_residual

    @property
    def last_ramp_progress(self) -> float:
        return self._last_ramp_progress

    @property
    def nnls_residual_history(self) -> list[float]:
        return list(self._nnls_residual_history)

    @property
    def ramp_progress_history(self) -> list[float]:
        return list(self._ramp_progress_history)

    @property
    def activation_history(self) -> list[np.ndarray]:
        return [a.copy() for a in self._activation_history]

    def reset(self, *, seed: int | None = None) -> None:
        del seed
        self._step = 0
        self._nnls_residual_history.clear()
        self._ramp_progress_history.clear()
        self._activation_history.clear()
        self._last_nnls_residual = 0.0
        self._last_ramp_progress = 0.0

    def act(self, observation: MyoArmState) -> np.ndarray:
        if not isinstance(observation, MyoArmState):
            raise ValueError(
                f"observation must be MyoArmState, "
                f"got {type(observation).__name__}"
            )
        # ramped virtual target
        if self._T_ramp <= 0:
            s = 1.0
        else:
            s = min(self._step / self._T_ramp, 1.0)
        cur_target = self._init_tip + s * (
            self._final_target - self._init_tip
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

        # task-space PD
        e_tip = cur_target - tip_est                          # (3,)
        v_tip = self._jacobian @ qvel                         # (3,)
        u_tip = self._Kp * e_tip - self._Kd * v_tip           # (3,)
        u_joint = self._jacobian.T @ u_tip                    # (nv,)

        # NNLS muscle routing: a >= 0, minimising ||M^T a - u_joint||.
        activation, residual = _scipy_nnls(
            self._mt, u_joint.astype(np.float64),
        )
        u_muscle = np.clip(
            self._action_scale * activation, _LO, _HI,
        ).astype(_DTYPE, copy=False)

        self._last_nnls_residual = float(residual)
        self._last_ramp_progress = float(s)
        if self._record_history:
            self._nnls_residual_history.append(float(residual))
            self._ramp_progress_history.append(float(s))
            self._activation_history.append(u_muscle.copy())

        self._step += 1
        return u_muscle

    def __repr__(self) -> str:
        return (
            f"StabilizedEndpointController(action_dim={self._action_dim}, "
            f"nv={self._nv}, Kp={self._Kp}, Kd={self._Kd}, "
            f"action_scale={self._action_scale}, T_ramp={self._T_ramp})"
        )
