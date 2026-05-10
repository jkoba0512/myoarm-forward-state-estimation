"""Smoke tests for run_episode + collect_episodes (require MyoSuite + MuJoCo).

Run with::

    uv run pytest -m myosuite
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("myosuite")

from myoarm_fse.controllers import HoldController, RandomController  # noqa: E402
from myoarm_fse.data import EpisodeLog, EpisodeSpec, RunIndex, run_episode  # noqa: E402
from myoarm_fse.envs.actions import ActionAdapter, detect_action_dim  # noqa: E402
from myoarm_fse.envs.extractors import extract_state  # noqa: E402
from myoarm_fse.envs.factory import make_env  # noqa: E402
from myoarm_fse.envs.noise import SignalDependentMotorNoise  # noqa: E402
from myoarm_fse.envs.targets import (  # noqa: E402
    SplitConfig,
    TargetGenerationConfig,
    generate_target_set,
)
from myoarm_fse.envs.wrappers import (  # noqa: E402
    DelayedObservationWrapper,
    NoisyObservationWrapper,
)

pytestmark = pytest.mark.myosuite


_ENV_ID = "myoArmReachFixed-v0"


@pytest.fixture
def env_and_components():
    env = make_env(_ENV_ID, horizon=20)
    try:
        env.reset()
        live_state = extract_state(env)
        state_spec = live_state.spec()
        action_dim = detect_action_dim(env)
        adapter = ActionAdapter(action_dim=action_dim)
        yield env, state_spec, action_dim, adapter
    finally:
        env.close()


@pytest.fixture
def small_target_set():
    cfg = TargetGenerationConfig(
        env_id="myoArmReachRandom-v0",
        generator_seed=0,
        output_dir="runs/targets",
        splits=(SplitConfig(name="train", n=1, seed_offset=0),),
    )
    return generate_target_set(cfg, "train")


# --- run_episode smoke ---


def test_short_episode_runs(env_and_components, small_target_set) -> None:
    env, spec, action_dim, adapter = env_and_components
    target_pos = small_target_set.target_pos[0]
    controller = RandomController(action_dim=action_dim, rng=0)

    log = run_episode(
        env, controller, target_pos,
        state_spec=spec, action_adapter=adapter, max_steps=10,
    )
    assert isinstance(log, EpisodeLog)
    assert log.n_steps > 0
    assert log.api_action.shape == (log.n_steps, action_dim)


def test_target_pos_matches_set_value(env_and_components, small_target_set) -> None:
    """Recorded true_target_pos at step 0 must equal the value we injected."""
    env, spec, _, adapter = env_and_components
    target_pos = small_target_set.target_pos[0]
    controller = RandomController(action_dim=adapter.action_dim, rng=0)

    log = run_episode(
        env, controller, target_pos,
        state_spec=spec, action_adapter=adapter, max_steps=3,
    )
    np.testing.assert_allclose(log.true_target_pos[0], target_pos, atol=1e-5)
    np.testing.assert_allclose(log.target_pos_set, target_pos, atol=1e-5)


def test_identity_pipeline_true_equals_obs(env_and_components, small_target_set) -> None:
    """No SDN, no noise, no delay → observed state must equal true state."""
    env, spec, action_dim, adapter = env_and_components
    target_pos = small_target_set.target_pos[0]
    controller = RandomController(action_dim=action_dim, rng=0)

    log = run_episode(
        env, controller, target_pos,
        state_spec=spec, action_adapter=adapter, max_steps=4,
    )
    np.testing.assert_array_equal(log.true_qpos, log.obs_qpos)
    np.testing.assert_array_equal(log.true_qvel, log.obs_qvel)
    np.testing.assert_array_equal(log.true_act, log.obs_act)
    np.testing.assert_array_equal(log.true_tip_pos, log.obs_tip_pos)
    np.testing.assert_array_equal(log.true_target_pos, log.obs_target_pos)
    np.testing.assert_array_equal(log.true_reach_err, log.obs_reach_err)


def test_dtype_is_float32(env_and_components, small_target_set) -> None:
    env, spec, action_dim, adapter = env_and_components
    controller = RandomController(action_dim=action_dim, rng=0)
    log = run_episode(
        env, controller, small_target_set.target_pos[0],
        state_spec=spec, action_adapter=adapter, max_steps=3,
    )
    for name in (
        "true_qpos", "true_qvel", "true_act",
        "obs_qpos", "obs_qvel", "obs_act",
        "neural_command", "excitation_command", "excitation",
        "api_action", "last_ctrl",
    ):
        assert getattr(log, name).dtype == np.float32, name


def test_with_sdn_excitation_perturbed(env_and_components, small_target_set) -> None:
    env, spec, action_dim, adapter = env_and_components
    controller = RandomController(action_dim=action_dim, rng=0)
    sdn = SignalDependentMotorNoise(action_dim=action_dim, sigma=0.5, rng=0)
    log = run_episode(
        env, controller, small_target_set.target_pos[0],
        state_spec=spec, action_adapter=adapter, sdn=sdn, max_steps=5,
    )
    assert log.sdn_sigma == pytest.approx(0.5)
    assert (np.abs(log.excitation - log.excitation_command) > 0).any()


def test_with_obs_noise_qpos_differs(env_and_components, small_target_set) -> None:
    env, spec, action_dim, adapter = env_and_components
    controller = RandomController(action_dim=action_dim, rng=0)
    obs_noise = NoisyObservationWrapper(spec=spec, sigma={"qpos": 0.5}, rng=0)
    log = run_episode(
        env, controller, small_target_set.target_pos[0],
        state_spec=spec, action_adapter=adapter,
        obs_noise=obs_noise, max_steps=5,
    )
    assert log.obs_noise_sigma == {"qpos": 0.5}
    assert not np.array_equal(log.true_qpos, log.obs_qpos)


def test_with_obs_delay_initial_observed(env_and_components, small_target_set) -> None:
    env, spec, action_dim, adapter = env_and_components
    controller = RandomController(action_dim=action_dim, rng=0)
    obs_delay = DelayedObservationWrapper(spec=spec, delay_steps=2)
    log = run_episode(
        env, controller, small_target_set.target_pos[0],
        state_spec=spec, action_adapter=adapter,
        obs_delay=obs_delay, max_steps=5,
    )
    assert log.obs_delay_steps == 2


def test_hold_controller_in_pipeline(env_and_components, small_target_set) -> None:
    """HoldController emits a constant excitation_command across all steps."""
    env, spec, action_dim, adapter = env_and_components
    controller = HoldController(action_dim=action_dim, value=0.5)
    log = run_episode(
        env, controller, small_target_set.target_pos[0],
        state_spec=spec, action_adapter=adapter, max_steps=5,
    )
    assert log.controller_name == "HoldController"
    np.testing.assert_array_equal(
        log.excitation_command,
        np.full((log.n_steps, action_dim), 0.5, dtype=np.float32),
    )
    np.testing.assert_array_equal(log.neural_command, log.excitation_command)


# --- save / load round trip on real data ---


def test_episode_save_load_round_trip(
    env_and_components, small_target_set, tmp_path: Path
) -> None:
    env, spec, action_dim, adapter = env_and_components
    controller = RandomController(action_dim=action_dim, rng=0)
    log = run_episode(
        env, controller, small_target_set.target_pos[0],
        state_spec=spec, action_adapter=adapter, max_steps=3,
        spec=EpisodeSpec(
            episode_id=0,
            target_id="train:0",
            target_split="train",
            target_seed=int(small_target_set.seeds[0]),
            controller_name="RandomController",
            controller_seed=0,
        ),
    )
    path = tmp_path / "0000.npz"
    log.save(path)
    loaded = EpisodeLog.load(path)
    np.testing.assert_array_equal(log.api_action, loaded.api_action)
    np.testing.assert_array_equal(log.true_qpos, loaded.true_qpos)
    assert loaded.target_id == "train:0"
    assert loaded.target_seed == int(small_target_set.seeds[0])


def test_run_index_round_trip_with_real_data(
    env_and_components, small_target_set, tmp_path: Path
) -> None:
    from myoarm_fse.data.logger import IndexEntry, RunIndex, hash_config, make_run_id

    env, spec, action_dim, adapter = env_and_components
    controller = RandomController(action_dim=action_dim, rng=0)

    run_id = make_run_id()
    out_dir = tmp_path / run_id
    out_dir.mkdir(parents=True)
    cfg = {"env_id": _ENV_ID, "max_steps": 3}
    index = RunIndex(
        run_id=run_id,
        created_at="now",
        config_hash=hash_config(cfg),
        config=cfg,
        target_set_path="(synthetic)",
    )

    log = run_episode(
        env, controller, small_target_set.target_pos[0],
        state_spec=spec, action_adapter=adapter, max_steps=3,
    )
    log.save(out_dir / "0000.npz")
    index.append(
        IndexEntry(
            episode_id=0, file="0000.npz",
            target_id="train:0",
            target_seed=int(small_target_set.seeds[0]),
            n_steps=log.n_steps,
        )
    )
    index.save(out_dir / "index.json")

    loaded = RunIndex.load(out_dir / "index.json")
    assert loaded.run_id == run_id
    assert len(loaded.episodes) == 1
    assert loaded.config_hash == hash_config(cfg)
