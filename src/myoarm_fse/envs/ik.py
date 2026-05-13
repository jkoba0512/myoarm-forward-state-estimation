"""Inverse-kinematics solver for myoArm target_pos -> target_qpos.

The myoArm has ``nq=20`` joint dofs and the reach task tip is the
``IFtip`` site (index-finger fingertip). The Cartesian target is 3-dim,
so the IK problem is heavily under-determined; we solve it with a
damped Jacobian-pseudoinverse iteration, which yields smooth minimum-
norm steps and converges in a few dozen iterations from the env's
neutral reset pose.

Side-effect contract: the solver writes ``mj_data.qpos`` and calls
``mj_forward`` during iteration. Callers that need to preserve the
env's pre-call state should snapshot ``mj_data.qpos / qvel`` before
``solve_ik`` and restore them afterwards (the helper below offers a
context manager for this).
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

import mujoco
import numpy as np

_CART_DIM: int = 3


@contextmanager
def _preserve_qpos_qvel(env: Any) -> Iterator[None]:
    """Snapshot qpos/qvel/ctrl, yield, then restore. Calls mj_forward at exit."""
    uw = env.unwrapped
    m = uw.mj_model
    d = uw.mj_data
    qpos_save = d.qpos.copy()
    qvel_save = d.qvel.copy()
    ctrl_save = d.ctrl.copy()
    try:
        yield
    finally:
        d.qpos[:] = qpos_save
        d.qvel[:] = qvel_save
        d.ctrl[:] = ctrl_save
        mujoco.mj_forward(m, d)


def solve_ik(
    env: Any,
    target_pos: np.ndarray,
    *,
    max_iter: int = 200,
    tol: float = 0.01,
    damping: float = 0.1,
    step_size: float = 1.0,
    preserve_env_state: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Damped Jacobian-pseudoinverse IK for the env's tip site.

    Parameters
    ----------
    env : gym.Env
        A MyoSuite reach env (``env.unwrapped.tip_sids`` must contain a
        single site id). The solver writes ``mj_data.qpos`` during
        iteration; pass ``preserve_env_state=True`` (default) to restore
        the env's pre-call qpos/qvel/ctrl afterwards.
    target_pos : np.ndarray
        Desired tip Cartesian position, shape ``(3,)``.
    max_iter : int
        Maximum Jacobian iterations.
    tol : float
        Stop early when ``||tip_pos - target_pos|| < tol`` (metres).
    damping : float
        Levenberg-Marquardt damping (added to ``J J^T`` diagonal).
        Larger values produce smaller per-step updates and stabler
        convergence at the cost of slower descent.
    step_size : float
        Multiplier on the Jacobian-pseudoinverse step (``1.0`` = full
        damped Newton step).

    Returns
    -------
    (qpos, info)
        ``qpos`` is a length-``nq`` ``np.float64`` copy of the joint
        configuration that places the tip closest to ``target_pos``.
        ``info`` reports ``{converged, final_error, n_iter}``.
    """
    target_arr = np.asarray(target_pos, dtype=np.float64)
    if target_arr.shape != (_CART_DIM,):
        raise ValueError(
            f"target_pos must have shape (3,), got {target_arr.shape}"
        )
    if not np.isfinite(target_arr).all():
        raise ValueError("target_pos contains non-finite values")
    if max_iter <= 0:
        raise ValueError(f"max_iter must be > 0, got {max_iter}")
    if tol <= 0:
        raise ValueError(f"tol must be > 0, got {tol}")
    if damping < 0:
        raise ValueError(f"damping must be >= 0, got {damping}")

    uw = env.unwrapped
    m = uw.mj_model
    d = uw.mj_data
    tip_sids = uw.tip_sids
    if len(tip_sids) != 1:
        raise ValueError(
            f"expected exactly one tip site, got tip_sids={tip_sids!r}"
        )
    tip_sid = int(tip_sids[0])

    # Joint limits (joint_range) for soft clipping. Use only the
    # ranges for limited joints; unlimited stay free.
    jnt_range = m.jnt_range.copy()
    jnt_limited = m.jnt_limited.astype(bool)

    def _maybe_preserve():
        return _preserve_qpos_qvel(env) if preserve_env_state else _null()

    @contextmanager
    def _null() -> Iterator[None]:
        yield

    final_err = float("inf")
    converged = False
    n_iter = 0
    qpos_result = d.qpos.copy()

    with _maybe_preserve():
        for it in range(max_iter):
            mujoco.mj_kinematics(m, d)
            tip_pos = d.site_xpos[tip_sid].copy()
            err_vec = target_arr - tip_pos
            err = float(np.linalg.norm(err_vec))
            final_err = err
            n_iter = it + 1
            if err < tol:
                converged = True
                qpos_result = d.qpos.copy()
                break

            jacp = np.zeros((_CART_DIM, m.nv), dtype=np.float64)
            jacr = np.zeros((_CART_DIM, m.nv), dtype=np.float64)
            mujoco.mj_jacSite(m, d, jacp, jacr, tip_sid)

            # Damped pseudoinverse: dq = J^T (J J^T + λ²I)^-1 err
            JJt = jacp @ jacp.T
            JJt += (damping ** 2) * np.eye(_CART_DIM)
            dq = jacp.T @ np.linalg.solve(JJt, err_vec)
            d.qpos[:] += step_size * dq

            # Soft clip to joint limits.
            # nq may not equal nv when there are free joints, but for
            # myoArm all joints are hinge so nq == nv (verified at probe).
            for j in range(m.njnt):
                if not jnt_limited[j]:
                    continue
                # Map joint j to its qpos start address. For hinge / slide
                # joints the qpos start equals the joint's qpos index.
                qpos_adr = m.jnt_qposadr[j]
                lo, hi = jnt_range[j]
                d.qpos[qpos_adr] = float(np.clip(d.qpos[qpos_adr], lo, hi))
            qpos_result = d.qpos.copy()

    return qpos_result, {
        "converged": bool(converged),
        "final_error": float(final_err),
        "n_iter": int(n_iter),
    }


def actuator_moment_dense(env: Any) -> np.ndarray:
    """Return the env's actuator moment matrix as dense ``(nu, nv)``.

    MyoSuite/MuJoCo stores ``actuator_moment`` in sparse CSR-ish form
    (``moment_rowadr``, ``moment_rownnz``, ``moment_colind``,
    ``actuator_moment``); we reconstruct the dense matrix at the env's
    current pose (caller is responsible for ``mj_forward`` if the pose
    matters).
    """
    uw = env.unwrapped
    m = uw.mj_model
    d = uw.mj_data
    nu = m.nu
    nv = m.nv
    out = np.zeros((nu, nv), dtype=np.float64)
    rowadr = d.moment_rowadr
    rownnz = d.moment_rownnz
    colind = d.moment_colind
    data = d.actuator_moment
    for i in range(nu):
        start = int(rowadr[i])
        n = int(rownnz[i])
        if n == 0:
            continue
        cols = colind[start : start + n]
        vals = data[start : start + n]
        out[i, cols] = vals
    return out


def tip_jacobian_dense(env: Any) -> np.ndarray:
    """Return the tip-site positional Jacobian as ``(3, nv)``.

    The Jacobian is computed at the env's current pose; caller is
    responsible for ``mj_forward`` if the pose matters. Used by the
    endpoint-error feedback controller to transform task-space (tip)
    commands into joint-space velocities via :math:`J^\\top`.
    """
    uw = env.unwrapped
    m = uw.mj_model
    d = uw.mj_data
    tip_sids = uw.tip_sids
    if len(tip_sids) != 1:
        raise ValueError(
            f"expected exactly one tip site, got tip_sids={tip_sids!r}"
        )
    tip_sid = int(tip_sids[0])
    jacp = np.zeros((_CART_DIM, m.nv), dtype=np.float64)
    jacr = np.zeros((_CART_DIM, m.nv), dtype=np.float64)
    mujoco.mj_jacSite(m, d, jacp, jacr, tip_sid)
    return jacp.copy()
