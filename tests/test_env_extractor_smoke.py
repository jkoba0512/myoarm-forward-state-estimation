"""Smoke tests for myoarm_fse.envs.extractors (require MyoSuite + MuJoCo).

Run with::

    uv run pytest -m myosuite
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("myosuite")

import gymnasium as gym  # noqa: E402

from myoarm_fse.envs import MyoArmState  # noqa: E402
from myoarm_fse.envs.extractors import extract_ctrl, extract_state  # noqa: E402
from myoarm_fse.envs.factory import make_env  # noqa: E402

pytestmark = pytest.mark.myosuite


_ENV_ID = "myoArmReachFixed-v0"


@pytest.fixture
def env() -> gym.Env:
    e = make_env(_ENV_ID)
    e.reset(seed=0)
    yield e
    e.close()


# --- extract_state ---


def test_extract_state_after_reset(env: gym.Env) -> None:
    state = extract_state(env)
    assert isinstance(state, MyoArmState)


def test_extract_state_dims_match_myoarm(env: gym.Env) -> None:
    """myoArm dims recorded for traceability (qpos=20, qvel=20, act=34)."""
    state = extract_state(env)
    assert state.qpos.shape == (20,)
    assert state.qvel.shape == (20,)
    assert state.act.shape == (34,)
    assert state.tip_pos.shape == (3,)
    assert state.target_pos.shape == (3,)
    assert state.reach_err.shape == (3,)


def test_extract_state_dtype_is_float32(env: gym.Env) -> None:
    state = extract_state(env)
    for name in ("qpos", "qvel", "act", "tip_pos", "target_pos", "reach_err"):
        assert getattr(state, name).dtype == np.float32


def test_extract_state_after_step(env: gym.Env) -> None:
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    env.step(action)
    state = extract_state(env)
    assert isinstance(state, MyoArmState)


def test_reach_err_uses_project_convention(env: gym.Env) -> None:
    """Project schema: reach_err = tip_pos - target_pos.

    MyoSuite's obs_dict uses the opposite sign (target - tip); the
    extractor must NOT pass that through.
    """
    state = extract_state(env)
    expected = state.tip_pos - state.target_pos
    np.testing.assert_allclose(state.reach_err, expected, atol=1e-6)
    # Also verify the sign disagrees with MyoSuite's obs_dict (sanity).
    od_reach_err = env.unwrapped.obs_dict["reach_err"]
    np.testing.assert_allclose(od_reach_err, state.target_pos - state.tip_pos, atol=1e-6)


def test_qvel_is_raw_not_dt_scaled(env: gym.Env) -> None:
    """obs_dict['qvel'] is mj_data.qvel * dt; extractor must use raw qvel."""
    # Take a step with a non-zero action so qvel is non-trivial.
    action = np.full(env.action_space.shape, 0.5, dtype=np.float32)
    env.step(action)
    state = extract_state(env)
    raw_qvel = env.unwrapped.mj_data.qvel.copy()
    np.testing.assert_allclose(state.qvel, raw_qvel.astype(np.float32), atol=1e-6)
    # Confirm obs_dict's qvel is the dt-scaled version (not what we use).
    dt_scaled = env.unwrapped.obs_dict["qvel"]
    np.testing.assert_allclose(
        dt_scaled, raw_qvel * env.unwrapped.dt, atol=1e-8
    )


def test_extract_state_returns_independent_arrays(env: gym.Env) -> None:
    """Mutating the returned state must not corrupt MuJoCo's data buffers."""
    state = extract_state(env)
    qpos_before = env.unwrapped.mj_data.qpos.copy()
    state.qpos[:] = 0.0  # safe because frozen=True only forbids reassignment
    qpos_after = env.unwrapped.mj_data.qpos.copy()
    np.testing.assert_array_equal(qpos_before, qpos_after)


# --- extract_ctrl ---


def test_extract_ctrl_shape_and_dtype(env: gym.Env) -> None:
    ctrl = extract_ctrl(env)
    assert ctrl.shape == (env.action_space.shape[0],)
    assert ctrl.dtype == np.float32


def test_extract_ctrl_returns_copy(env: gym.Env) -> None:
    ctrl = extract_ctrl(env)
    ctrl[:] = 999.0
    assert not np.any(env.unwrapped.last_ctrl == 999.0)


def test_extract_ctrl_reflects_step_input(env: gym.Env) -> None:
    """After step(api_action) with normalize_act=True, last_ctrl is
    sigmoid(api_action) over [0, 1]; api_action=0 maps to ~0.5."""
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    env.step(action)
    ctrl = extract_ctrl(env)
    # sigmoid(5*(0 - 0.5))^{-1} ≈ sigmoid(-2.5) ≈ 0.0759 — actually
    # MyoSuite uses 1 / (1 + exp(-5*(a-0.5))). For a=0 that's
    # 1/(1+exp(2.5)) ≈ 0.0759.
    expected = 1.0 / (1.0 + np.exp(-5.0 * (action - 0.5)))
    np.testing.assert_allclose(ctrl, expected.astype(np.float32), atol=1e-5)
