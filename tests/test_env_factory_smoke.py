"""Smoke tests for myoarm_fse.envs.factory.make_env (require MyoSuite + MuJoCo).

Run with::

    uv run pytest -m myosuite

Default ``uv run pytest`` skips this file via the ``-m 'not myosuite'``
addopts configured in ``pyproject.toml``.
"""

from __future__ import annotations

import pytest

pytest.importorskip("myosuite")

import gymnasium as gym  # noqa: E402

from myoarm_fse.envs.factory import make_env  # noqa: E402

pytestmark = pytest.mark.myosuite


_ENV_ID = "myoArmReachFixed-v0"


@pytest.fixture
def env() -> gym.Env:
    e = make_env(_ENV_ID)
    yield e
    e.close()


def test_make_env_returns_gym_env(env: gym.Env) -> None:
    assert isinstance(env, gym.Env)


def test_horizon_pinned_to_default(env: gym.Env) -> None:
    assert env.spec.max_episode_steps == 600
    assert env.unwrapped.horizon == 600


def test_horizon_override() -> None:
    e = make_env(_ENV_ID, horizon=200)
    try:
        assert e.spec.max_episode_steps == 200
        assert e.unwrapped.horizon == 200
    finally:
        e.close()


def test_normalize_act_pinned_default(env: gym.Env) -> None:
    assert env.unwrapped.normalize_act is True
    assert (env.action_space.low == -1.0).all()
    assert (env.action_space.high == 1.0).all()


def test_normalize_act_override() -> None:
    e = make_env(_ENV_ID, normalize_act=False)
    try:
        assert e.unwrapped.normalize_act is False
        # When normalize_act=False the action_space tracks the actuator range
        # ([0, 1] for muscles). We don't assert exact bounds here — only that
        # it differs from the normalized [-1, 1] case.
        assert not (e.action_space.low == -1.0).all()
    finally:
        e.close()


def test_action_dim_detected(env: gym.Env) -> None:
    from myoarm_fse.envs import detect_action_dim

    # myoArm has 34 muscle actuators; this is a smoke check of the value
    # used throughout the project (see also Step 3 layer notes).
    assert detect_action_dim(env) == 34


def test_obs_shape_is_80(env: gym.Env) -> None:
    # Recorded for traceability: raw Gym observation_space.shape for myoArm
    # reach is (80,). If MyoSuite changes this, downstream extractor
    # assumptions need re-validation.
    assert env.observation_space.shape == (80,)


# --- constructor validation ---


@pytest.mark.parametrize("bad", [0, -1])
def test_non_positive_horizon_raises(bad: int) -> None:
    with pytest.raises(ValueError):
        make_env(_ENV_ID, horizon=bad)


def test_bool_horizon_raises() -> None:
    with pytest.raises(ValueError):
        make_env(_ENV_ID, horizon=True)  # type: ignore[arg-type]


def test_float_horizon_raises() -> None:
    with pytest.raises(ValueError):
        make_env(_ENV_ID, horizon=600.0)  # type: ignore[arg-type]


def test_non_bool_normalize_act_raises() -> None:
    with pytest.raises(TypeError):
        make_env(_ENV_ID, normalize_act=1)  # type: ignore[arg-type]
