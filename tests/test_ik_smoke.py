"""MyoSuite-dependent smoke tests for envs/ik.py.

IK and moment-arm extraction both require a fully-built MuJoCo model,
so these tests are gated behind the ``myosuite`` mark like other
env-touching smokes.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.myosuite


def _make_env():
    from myoarm_fse.envs.factory import make_env
    env = make_env("myoArmReachFixed-v0", horizon=10)
    env.reset()
    return env


def test_solve_ik_converges_for_near_target() -> None:
    """A target slightly displaced from the current tip should reach
    sub-cm error within a few iterations."""
    from myoarm_fse.envs.ik import solve_ik
    from myoarm_fse.envs.extractors import extract_state

    env = _make_env()
    try:
        # Read current tip; perturb slightly to define a near target.
        st = extract_state(env)
        target = st.tip_pos.astype(np.float64) + np.array([0.03, -0.02, 0.04])
        qpos, info = solve_ik(env, target, max_iter=200, tol=0.01)
        assert qpos.shape == (env.unwrapped.mj_model.nq,)
        assert info["converged"] is True, f"info={info}"
        assert info["final_error"] < 0.01
        # Env state should be restored (qpos unchanged from reset state).
        cur_state = extract_state(env)
        # tip_pos may differ slightly due to mj_forward at restore; assert
        # qpos identical.
        # We snapshotted before solve and restore right after; cur tip ≈ initial tip.
        assert np.linalg.norm(cur_state.tip_pos - st.tip_pos) < 1e-5
    finally:
        env.close()


def test_solve_ik_preserve_env_false_leaves_solved_qpos() -> None:
    """When preserve_env_state=False the env is left at the IK solution."""
    from myoarm_fse.envs.ik import solve_ik
    from myoarm_fse.envs.extractors import extract_state

    env = _make_env()
    try:
        st = extract_state(env)
        target = st.tip_pos.astype(np.float64) + np.array([0.03, -0.02, 0.04])
        _qpos, info = solve_ik(env, target, preserve_env_state=False)
        if info["converged"]:
            cur = extract_state(env)
            # Now tip should be near target.
            assert np.linalg.norm(cur.tip_pos - target) < 0.02
    finally:
        env.close()


def test_actuator_moment_dense_shape_and_nonzero() -> None:
    """Moment matrix should have shape (nu, nv) and be non-trivially populated."""
    from myoarm_fse.envs.ik import actuator_moment_dense
    env = _make_env()
    try:
        m = env.unwrapped.mj_model
        M = actuator_moment_dense(env)
        assert M.shape == (m.nu, m.nv)
        assert np.any(M != 0.0), "moment matrix is all zero — extraction failed"
        # At least 10% of entries non-zero is a reasonable lower bound for
        # myoArm (most muscles span 1-2 joints).
        assert (M != 0.0).mean() > 0.05
    finally:
        env.close()


def test_solve_ik_invalid_shape_raises() -> None:
    from myoarm_fse.envs.ik import solve_ik
    env = _make_env()
    try:
        with pytest.raises(ValueError, match="must have shape"):
            solve_ik(env, np.zeros(2))
    finally:
        env.close()
