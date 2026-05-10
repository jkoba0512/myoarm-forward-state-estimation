"""Unit tests for myoarm_fse.data.rollout.run_episode (no MyoSuite required).

Uses a minimal fake env that mimics the env.unwrapped surface our
extractor and target-injection code rely on. ``mujoco.mj_forward`` is
monkeypatched to a no-op-ish ``site_pos -> site_xpos`` copy so we can
test the rollout pipeline end-to-end without spinning up MuJoCo.
"""

from __future__ import annotations

import numpy as np
import pytest

from myoarm_fse.controllers.random import RandomController
from myoarm_fse.data.rollout import EpisodeSpec, run_episode
from myoarm_fse.envs.actions import ActionAdapter
from myoarm_fse.envs.noise import SignalDependentMotorNoise
from myoarm_fse.envs.state import StateSpec
from myoarm_fse.envs.wrappers import (
    DelayedObservationWrapper,
    NoisyObservationWrapper,
)


# --- fake env infrastructure ---


class _FakeMjData:
    def __init__(self, qpos_dim: int, qvel_dim: int, act_dim: int) -> None:
        self.qpos = np.zeros(qpos_dim, dtype=np.float64)
        self.qvel = np.zeros(qvel_dim, dtype=np.float64)
        self.act = np.zeros(act_dim, dtype=np.float64)
        self.site_xpos = np.zeros((2, 3), dtype=np.float64)
        self.ctrl = np.zeros(act_dim, dtype=np.float64)


class _FakeMjModel:
    def __init__(self) -> None:
        self.site_pos = np.zeros((2, 3), dtype=np.float64)


class _FakeUnwrapped:
    def __init__(
        self,
        action_dim: int = 4,
        qpos_dim: int = 2,
        qvel_dim: int = 3,
        act_dim: int = 4,
    ) -> None:
        self.mj_data = _FakeMjData(qpos_dim, qvel_dim, act_dim)
        self.mj_model = _FakeMjModel()
        self.tip_sids = [0]
        self.target_sids = [1]
        self.dt = 0.02
        self.last_ctrl = np.zeros(action_dim, dtype=np.float64)


class _FakeEnv:
    def __init__(
        self,
        action_dim: int = 4,
        qpos_dim: int = 2,
        qvel_dim: int = 3,
        act_dim: int = 4,
        terminate_at: int | None = None,
        truncate_at: int | None = None,
    ) -> None:
        self.unwrapped = _FakeUnwrapped(action_dim, qpos_dim, qvel_dim, act_dim)
        self._step = 0
        self._terminate_at = terminate_at
        self._truncate_at = truncate_at

    def reset(self) -> tuple:
        self._step = 0
        # Reset internal state.
        self.unwrapped.mj_data.qpos[:] = 0.0
        self.unwrapped.mj_data.qvel[:] = 0.0
        self.unwrapped.mj_data.act[:] = 0.0
        self.unwrapped.last_ctrl[:] = 0.0
        return None, {}

    def step(self, action: np.ndarray) -> tuple:
        self.unwrapped.last_ctrl[:] = np.asarray(action, dtype=np.float64)
        # Trivial pseudo-dynamics so qpos/qvel are non-zero across steps.
        self.unwrapped.mj_data.qpos[:] = (self._step + 1) * 0.01
        self.unwrapped.mj_data.qvel[:] = (self._step + 1) * 0.001
        self.unwrapped.mj_data.act[:] = np.clip(
            self.unwrapped.mj_data.act + 0.1, 0.0, 1.0
        )
        self._step += 1
        terminated = (
            self._terminate_at is not None and self._step >= self._terminate_at
        )
        truncated = self._truncate_at is not None and self._step >= self._truncate_at
        reward = -0.1
        return None, reward, terminated, truncated, {}


@pytest.fixture(autouse=True)
def patch_mj_forward(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace mujoco.mj_forward with a site_pos -> site_xpos copy."""
    import mujoco

    def fake_forward(model: object, data: object) -> None:
        # Mirror the parts that our extractor reads.
        data.site_xpos[:] = model.site_pos[:]

    monkeypatch.setattr(mujoco, "mj_forward", fake_forward)


# --- helpers ---


def _make_components(action_dim: int = 4) -> tuple:
    env = _FakeEnv(action_dim=action_dim, act_dim=action_dim)
    state_spec = StateSpec(qpos_dim=2, qvel_dim=3, act_dim=action_dim)
    action_adapter = ActionAdapter(action_dim=action_dim)
    controller = RandomController(action_dim=action_dim, rng=0)
    target_pos = np.array([0.5, -0.3, 1.2], dtype=np.float32)
    return env, state_spec, action_adapter, controller, target_pos


# --- basic rollout ---


def test_basic_rollout_returns_episode_log() -> None:
    env, spec, adapter, controller, target = _make_components()
    log = run_episode(
        env, controller, target,
        state_spec=spec, action_adapter=adapter, max_steps=5,
    )
    assert log.n_steps == 5
    assert log.max_steps == 5
    assert log.api_action.shape == (5, 4)
    assert log.true_qpos.shape == (5, 2)
    assert log.true_qvel.shape == (5, 3)
    assert log.true_act.shape == (5, 4)
    assert log.true_tip_pos.shape == (5, 3)
    assert log.target_pos_set.tolist() == pytest.approx([0.5, -0.3, 1.2], rel=1e-5)


def test_target_injected_via_site_pos() -> None:
    env, spec, adapter, controller, target = _make_components()
    log = run_episode(
        env, controller, target,
        state_spec=spec, action_adapter=adapter, max_steps=3,
    )
    # After our fake mj_forward, target site_xpos[1] == target_pos.
    np.testing.assert_allclose(
        log.true_target_pos[0], target.astype(np.float32), atol=1e-6
    )


def test_reach_err_is_tip_minus_target() -> None:
    env, spec, adapter, controller, target = _make_components()
    log = run_episode(
        env, controller, target,
        state_spec=spec, action_adapter=adapter, max_steps=3,
    )
    np.testing.assert_allclose(
        log.true_reach_err, log.true_tip_pos - log.true_target_pos, atol=1e-6
    )


def test_termination_truncates_log() -> None:
    env = _FakeEnv(action_dim=4, act_dim=4, terminate_at=3)
    spec = StateSpec(qpos_dim=2, qvel_dim=3, act_dim=4)
    adapter = ActionAdapter(action_dim=4)
    controller = RandomController(action_dim=4, rng=0)
    log = run_episode(
        env, controller, np.zeros(3, dtype=np.float32),
        state_spec=spec, action_adapter=adapter, max_steps=10,
    )
    assert log.n_steps == 3
    assert log.terminated[-1] is np.True_ or bool(log.terminated[-1]) is True
    assert log.api_action.shape == (3, 4)


def test_truncation_truncates_log() -> None:
    env = _FakeEnv(action_dim=4, act_dim=4, truncate_at=2)
    spec = StateSpec(qpos_dim=2, qvel_dim=3, act_dim=4)
    adapter = ActionAdapter(action_dim=4)
    controller = RandomController(action_dim=4, rng=0)
    log = run_episode(
        env, controller, np.zeros(3, dtype=np.float32),
        state_spec=spec, action_adapter=adapter, max_steps=10,
    )
    assert log.n_steps == 2
    assert bool(log.truncated[-1]) is True


# --- identity wrappers ---


def test_identity_obs_pipeline_true_equals_obs() -> None:
    """With no SDN, no obs noise, no delay, observed must equal true."""
    env, spec, adapter, controller, target = _make_components()
    log = run_episode(
        env, controller, target,
        state_spec=spec, action_adapter=adapter, max_steps=4,
    )
    np.testing.assert_array_equal(log.true_qpos, log.obs_qpos)
    np.testing.assert_array_equal(log.true_qvel, log.obs_qvel)
    np.testing.assert_array_equal(log.true_act, log.obs_act)
    np.testing.assert_array_equal(log.true_tip_pos, log.obs_tip_pos)
    np.testing.assert_array_equal(log.true_target_pos, log.obs_target_pos)
    np.testing.assert_array_equal(log.true_reach_err, log.obs_reach_err)


def test_identity_excitation_chain() -> None:
    """No SDN: excitation == clip(excitation_command); api_action = 2*exc - 1."""
    env, spec, adapter, controller, target = _make_components()
    log = run_episode(
        env, controller, target,
        state_spec=spec, action_adapter=adapter, max_steps=3,
    )
    np.testing.assert_allclose(log.excitation, log.excitation_command, atol=1e-6)
    np.testing.assert_allclose(
        log.api_action, 2.0 * log.excitation - 1.0, atol=1e-6
    )


def test_neural_command_equals_excitation_command() -> None:
    """Current minimum design: identity translation."""
    env, spec, adapter, controller, target = _make_components()
    log = run_episode(
        env, controller, target,
        state_spec=spec, action_adapter=adapter, max_steps=3,
    )
    np.testing.assert_array_equal(log.neural_command, log.excitation_command)


# --- wrappers active ---


def test_with_sdn_excitation_differs_from_command() -> None:
    env, spec, adapter, controller, target = _make_components()
    sdn = SignalDependentMotorNoise(action_dim=4, sigma=0.5, rng=0)
    log = run_episode(
        env, controller, target,
        state_spec=spec, action_adapter=adapter, sdn=sdn, max_steps=4,
    )
    # With sigma>0 and non-zero excitation_command, SDN should perturb.
    diff = np.abs(log.excitation - log.excitation_command)
    assert (diff > 0).any()
    assert log.sdn_sigma == pytest.approx(0.5)


def test_with_obs_noise_obs_qpos_differs_from_true() -> None:
    env, spec, adapter, controller, target = _make_components()
    obs_noise = NoisyObservationWrapper(
        spec=spec, sigma={"qpos": 0.5}, rng=0
    )
    log = run_episode(
        env, controller, target,
        state_spec=spec, action_adapter=adapter,
        obs_noise=obs_noise, max_steps=4,
    )
    assert not np.array_equal(log.true_qpos, log.obs_qpos)
    np.testing.assert_array_equal(log.true_qvel, log.obs_qvel)
    assert log.obs_noise_sigma == {"qpos": 0.5}


def test_with_obs_delay_first_obs_is_initial() -> None:
    env, spec, adapter, controller, target = _make_components()
    obs_delay = DelayedObservationWrapper(spec=spec, delay_steps=2)
    log = run_episode(
        env, controller, target,
        state_spec=spec, action_adapter=adapter,
        obs_delay=obs_delay, max_steps=4,
    )
    assert log.obs_delay_steps == 2
    # delay_steps=2 means the first 2 observed states equal the initial true
    # state (which we recorded by stepping qpos from 0 to 0.01 inside step()).
    # The observed at t=0 reflects the *post-reset* qpos values (all zero).
    np.testing.assert_allclose(log.obs_qpos[0], 0.0, atol=1e-6)
    np.testing.assert_allclose(log.obs_qpos[1], 0.0, atol=1e-6)


# --- composition ---


def test_obs_compose_invalid_raises() -> None:
    env, spec, adapter, controller, target = _make_components()
    with pytest.raises(ValueError, match="obs_compose"):
        run_episode(
            env, controller, target,
            state_spec=spec, action_adapter=adapter,
            obs_compose="banana", max_steps=2,
        )


def test_obs_compose_default_is_noisy_then_delayed() -> None:
    # Smoke check that default doesn't raise.
    env, spec, adapter, controller, target = _make_components()
    log = run_episode(
        env, controller, target,
        state_spec=spec, action_adapter=adapter, max_steps=2,
    )
    assert log.obs_compose == "noisy_then_delayed"


# --- spec & metadata ---


def test_spec_metadata_recorded() -> None:
    env, spec_st, adapter, controller, target = _make_components()
    spec = EpisodeSpec(
        episode_id=7,
        target_id="train:42",
        target_split="train",
        target_seed=42,
        controller_name="random",
        controller_seed=99,
        sdn_seed=1,
        obs_noise_seed=2,
        config_hash="deadbeef",
        meta={"note": "test"},
    )
    log = run_episode(
        env, controller, target,
        state_spec=spec_st, action_adapter=adapter, max_steps=2, spec=spec,
    )
    assert log.episode_id == 7
    assert log.target_id == "train:42"
    assert log.target_seed == 42
    assert log.controller_name == "random"
    assert log.controller_seed == 99
    assert log.sdn_seed == 1
    assert log.obs_noise_seed == 2
    assert log.config_hash == "deadbeef"
    assert log.meta == {"note": "test"}


def test_state_spec_mismatch_raises() -> None:
    env = _FakeEnv(action_dim=4, qpos_dim=2, qvel_dim=3, act_dim=4)
    wrong_spec = StateSpec(qpos_dim=99, qvel_dim=3, act_dim=4)
    adapter = ActionAdapter(action_dim=4)
    controller = RandomController(action_dim=4, rng=0)
    with pytest.raises(ValueError, match="state_spec mismatch"):
        run_episode(
            env, controller, np.zeros(3, dtype=np.float32),
            state_spec=wrong_spec, action_adapter=adapter, max_steps=2,
        )


def test_target_pos_wrong_shape_raises() -> None:
    env, spec, adapter, controller, _ = _make_components()
    bad = np.zeros(2, dtype=np.float32)
    with pytest.raises(ValueError, match=r"shape \(3,\)"):
        run_episode(
            env, controller, bad,
            state_spec=spec, action_adapter=adapter, max_steps=2,
        )


def test_max_steps_zero_returns_empty_log() -> None:
    env, spec, adapter, controller, target = _make_components()
    log = run_episode(
        env, controller, target,
        state_spec=spec, action_adapter=adapter, max_steps=0,
    )
    assert log.n_steps == 0
    assert log.api_action.shape == (0, 4)
    assert log.reward.shape == (0,)
