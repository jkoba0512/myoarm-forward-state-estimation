"""Tests for myoarm_fse.models.datasets (no MyoSuite required)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from myoarm_fse.data import EpisodeLog
from myoarm_fse.models import (
    TransitionDataset,
    build_transitions,
    shuffle_transitions,
    split_by_episode,
)


# --- helpers ---


def _make_log(
    *,
    episode_id: int = 0,
    n_steps: int = 5,
    qpos_dim: int = 2,
    qvel_dim: int = 3,
    act_dim: int = 4,
    action_dim: int = 4,
    target_id: str | None = None,
    target_seed: int = 0,
    sdn_sigma: float = 0.0,
    obs_noise_sigma: dict | None = None,
    obs_delay_steps: int = 0,
    terminated_at: int | None = None,
    truncated_at: int | None = None,
    qpos_per_step: np.ndarray | None = None,
    excitation_per_step: np.ndarray | None = None,
) -> EpisodeLog:
    """Construct a minimal EpisodeLog with deterministic content.

    By default qpos[t] = t * 0.1 and excitation[t] = t * 0.01, so
    Δx is non-trivial and easy to predict in tests.
    """
    T = n_steps
    f32 = np.float32

    if qpos_per_step is None and T > 0:
        qpos = np.full((T, qpos_dim), 0.0, dtype=f32)
        for t in range(T):
            qpos[t] = t * 0.1
    else:
        qpos = (
            qpos_per_step.astype(f32)
            if qpos_per_step is not None
            else np.zeros((T, qpos_dim), dtype=f32)
        )

    if excitation_per_step is None:
        excitation = np.zeros((T, action_dim), dtype=f32)
        for t in range(T):
            excitation[t] = t * 0.01
    else:
        excitation = excitation_per_step.astype(f32)

    terminated = np.zeros(T, dtype=np.bool_)
    truncated = np.zeros(T, dtype=np.bool_)
    if terminated_at is not None and 0 <= terminated_at < T:
        terminated[terminated_at] = True
    if truncated_at is not None and 0 <= truncated_at < T:
        truncated[truncated_at] = True

    return EpisodeLog(
        episode_id=episode_id,
        target_id=target_id if target_id is not None else f"train:{episode_id}",
        target_split="train",
        target_seed=target_seed,
        target_pos_set=np.zeros(3, dtype=f32),
        controller_name="random",
        controller_seed=0,
        sdn_sigma=sdn_sigma,
        sdn_seed=0,
        obs_noise_sigma=obs_noise_sigma if obs_noise_sigma is not None else {},
        obs_noise_seed=0,
        obs_delay_steps=obs_delay_steps,
        obs_compose="noisy_then_delayed",
        max_steps=max(T, 1),
        n_steps=T,
        created_at="2026-05-10T00:00:00Z",
        config_hash="test",
        step=np.arange(T, dtype=np.int64),
        time=np.arange(T, dtype=f32) * 0.02,
        true_qpos=qpos,
        true_qvel=np.zeros((T, qvel_dim), dtype=f32),
        true_act=np.zeros((T, act_dim), dtype=f32),
        true_tip_pos=np.zeros((T, 3), dtype=f32),
        true_target_pos=np.zeros((T, 3), dtype=f32),
        true_reach_err=np.zeros((T, 3), dtype=f32),
        obs_qpos=qpos.copy(),
        obs_qvel=np.zeros((T, qvel_dim), dtype=f32),
        obs_act=np.zeros((T, act_dim), dtype=f32),
        obs_tip_pos=np.zeros((T, 3), dtype=f32),
        obs_target_pos=np.zeros((T, 3), dtype=f32),
        obs_reach_err=np.zeros((T, 3), dtype=f32),
        neural_command=excitation.copy(),
        excitation_command=excitation.copy(),
        excitation=excitation,
        api_action=np.zeros((T, action_dim), dtype=f32),
        last_ctrl=np.zeros((T, action_dim), dtype=f32),
        reward=np.zeros(T, dtype=f32),
        terminated=terminated,
        truncated=truncated,
    )


# --- TransitionDataset construction ---


def _good_dataset(N: int = 4, state_dim: int = 5, action_dim: int = 3) -> TransitionDataset:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((N, state_dim)).astype(np.float32)
    u = rng.standard_normal((N, action_dim)).astype(np.float32)
    x_next = x + rng.standard_normal((N, state_dim)).astype(np.float32) * 0.01
    dx = (x_next - x).astype(np.float32)
    return TransitionDataset(
        x=x,
        u=u,
        x_next=x_next,
        dx=dx,
        episode_index=np.zeros(N, dtype=np.int64),
        state_dim=state_dim,
        action_dim=action_dim,
        n_episodes=1,
        episode_metadata=({"episode_id": 0, "n_steps": N + 1},),
    )


class TestConstruction:
    def test_valid(self) -> None:
        ds = _good_dataset()
        assert ds.n == 4

    def test_empty(self) -> None:
        ds = TransitionDataset(
            x=np.empty((0, 5), dtype=np.float32),
            u=np.empty((0, 3), dtype=np.float32),
            x_next=np.empty((0, 5), dtype=np.float32),
            dx=np.empty((0, 5), dtype=np.float32),
            episode_index=np.empty((0,), dtype=np.int64),
            state_dim=5,
            action_dim=3,
            n_episodes=0,
            episode_metadata=(),
        )
        assert ds.n == 0

    def test_wrong_dtype_x(self) -> None:
        rng = np.random.default_rng(0)
        ds_kwargs = _dataset_kwargs(_good_dataset())
        ds_kwargs["x"] = ds_kwargs["x"].astype(np.float64)
        # dx consistency would still hold but float32 dtype check fires first
        with pytest.raises(ValueError, match="float32"):
            TransitionDataset(**ds_kwargs)

    def test_wrong_episode_index_dtype(self) -> None:
        ds_kwargs = _dataset_kwargs(_good_dataset())
        ds_kwargs["episode_index"] = ds_kwargs["episode_index"].astype(np.int32)
        with pytest.raises(ValueError, match="int64"):
            TransitionDataset(**ds_kwargs)

    def test_episode_index_out_of_range(self) -> None:
        ds_kwargs = _dataset_kwargs(_good_dataset())
        bad = ds_kwargs["episode_index"].copy()
        bad[0] = 99
        ds_kwargs["episode_index"] = bad
        with pytest.raises(ValueError, match=r"\[0, 1\)"):
            TransitionDataset(**ds_kwargs)

    def test_dx_inconsistent(self) -> None:
        ds_kwargs = _dataset_kwargs(_good_dataset())
        bad_dx = ds_kwargs["dx"].copy()
        bad_dx[0, 0] += 1.0
        ds_kwargs["dx"] = bad_dx
        with pytest.raises(ValueError, match="dx must equal"):
            TransitionDataset(**ds_kwargs)

    def test_nan_in_x(self) -> None:
        ds_kwargs = _dataset_kwargs(_good_dataset())
        ds_kwargs["x"] = ds_kwargs["x"].copy()
        ds_kwargs["x"][0, 0] = np.nan
        with pytest.raises(ValueError, match="non-finite"):
            TransitionDataset(**ds_kwargs)

    def test_metadata_length_mismatch(self) -> None:
        ds_kwargs = _dataset_kwargs(_good_dataset())
        ds_kwargs["episode_metadata"] = ()
        with pytest.raises(ValueError, match="episode_metadata length"):
            TransitionDataset(**ds_kwargs)

    def test_metadata_not_tuple(self) -> None:
        ds_kwargs = _dataset_kwargs(_good_dataset())
        ds_kwargs["episode_metadata"] = [{"episode_id": 0, "n_steps": 1}]
        with pytest.raises(ValueError, match="must be tuple"):
            TransitionDataset(**ds_kwargs)

    def test_metadata_must_be_dict(self) -> None:
        ds_kwargs = _dataset_kwargs(_good_dataset())
        ds_kwargs["episode_metadata"] = ("not a dict",)
        with pytest.raises(ValueError, match=r"episode_metadata\[0\] must be dict"):
            TransitionDataset(**ds_kwargs)

    def test_metadata_must_be_json_serializable(self) -> None:
        ds_kwargs = _dataset_kwargs(_good_dataset())
        ds_kwargs["episode_metadata"] = ({"x": object()},)
        with pytest.raises(ValueError, match="JSON-serializable"):
            TransitionDataset(**ds_kwargs)

    def test_negative_state_dim_rejected(self) -> None:
        ds_kwargs = _dataset_kwargs(_good_dataset())
        ds_kwargs["state_dim"] = -1
        with pytest.raises(ValueError):
            TransitionDataset(**ds_kwargs)

    def test_bool_state_dim_rejected(self) -> None:
        ds_kwargs = _dataset_kwargs(_good_dataset())
        ds_kwargs["state_dim"] = True  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            TransitionDataset(**ds_kwargs)


def _dataset_kwargs(ds: TransitionDataset) -> dict[str, Any]:
    from dataclasses import fields

    return {f.name: getattr(ds, f.name) for f in fields(ds)}


# --- save / load ---


class TestSaveLoad:
    def test_round_trip(self, tmp_path: Path) -> None:
        ds = _good_dataset()
        path = tmp_path / "ds.npz"
        ds.save(path)
        loaded = TransitionDataset.load(path)
        np.testing.assert_array_equal(ds.x, loaded.x)
        np.testing.assert_array_equal(ds.u, loaded.u)
        np.testing.assert_array_equal(ds.x_next, loaded.x_next)
        np.testing.assert_array_equal(ds.dx, loaded.dx)
        np.testing.assert_array_equal(ds.episode_index, loaded.episode_index)
        assert loaded.state_dim == ds.state_dim
        assert loaded.action_dim == ds.action_dim
        assert loaded.n_episodes == ds.n_episodes
        assert loaded.episode_metadata == ds.episode_metadata

    def test_save_creates_parent(self, tmp_path: Path) -> None:
        ds = _good_dataset()
        path = tmp_path / "deep" / "subdir" / "ds.npz"
        ds.save(path)
        assert path.exists()

    def test_load_disallows_pickle(self, tmp_path: Path) -> None:
        ds = _good_dataset()
        path = tmp_path / "ds.npz"
        ds.save(path)
        with np.load(path, allow_pickle=False) as f:
            assert "x" in f
            assert "meta_json" in f


# --- build_transitions ---


class TestBuildTransitions:
    def test_basic_single_episode(self) -> None:
        log = _make_log(episode_id=0, n_steps=5)
        ds = build_transitions([log])
        assert ds.n == 4  # T - 1
        assert ds.n_episodes == 1
        # state_dim = qpos(2) + qvel(3) + act(4) + 3*3 = 18
        assert ds.state_dim == 18
        assert ds.action_dim == 4
        np.testing.assert_array_equal(
            ds.episode_index, np.zeros(4, dtype=np.int64)
        )

    def test_multi_episode(self) -> None:
        logs = [_make_log(episode_id=i, n_steps=5) for i in range(3)]
        ds = build_transitions(logs)
        assert ds.n == 12  # 3 * 4
        assert ds.n_episodes == 3
        # episode_index should be [0,0,0,0,1,1,1,1,2,2,2,2]
        np.testing.assert_array_equal(
            ds.episode_index,
            np.array([0]*4 + [1]*4 + [2]*4, dtype=np.int64),
        )

    def test_skip_n_steps_lt_2(self) -> None:
        logs = [
            _make_log(episode_id=0, n_steps=5),
            _make_log(episode_id=1, n_steps=1),  # too short
            _make_log(episode_id=2, n_steps=4),
        ]
        ds = build_transitions(logs)
        assert ds.n_episodes == 2
        assert ds.n == 4 + 3
        # Source episode_id values preserved in metadata.
        ids = [m["episode_id"] for m in ds.episode_metadata]
        assert ids == [0, 2]

    def test_empty_input(self) -> None:
        ds = build_transitions([])
        assert ds.n == 0
        assert ds.n_episodes == 0
        assert ds.state_dim == 0
        assert ds.action_dim == 0

    def test_all_episodes_too_short(self) -> None:
        ds = build_transitions([_make_log(n_steps=1) for _ in range(3)])
        assert ds.n == 0
        assert ds.n_episodes == 0

    def test_mid_episode_termination_raises(self) -> None:
        log = _make_log(n_steps=5, terminated_at=2)
        with pytest.raises(ValueError, match="terminated/truncated"):
            build_transitions([log])

    def test_mid_episode_truncation_raises(self) -> None:
        log = _make_log(n_steps=5, truncated_at=1)
        with pytest.raises(ValueError, match="terminated/truncated"):
            build_transitions([log])

    def test_terminal_at_last_step_ok(self) -> None:
        # terminated set on the last step is the normal "early termination"
        # case from rollout — not a violation.
        log = _make_log(n_steps=5, terminated_at=4)
        ds = build_transitions([log])
        assert ds.n == 4

    def test_uses_excitation_not_neural_command(self) -> None:
        # Build a log where excitation differs from neural_command/excitation_command.
        log = _make_log(episode_id=0, n_steps=4)
        # Manually mutate excitation to confirm build_transitions reads it.
        # We bypass frozen=True by reconstructing via dataclasses.replace
        # on the underlying ndarray view — but EpisodeLog is frozen, so we
        # build a fresh log with excitation set differently.
        T = 4
        rng = np.random.default_rng(0)
        custom_exc = rng.standard_normal((T, 4)).astype(np.float32) * 0.01 + 0.5
        log = _make_log(
            episode_id=0,
            n_steps=T,
            excitation_per_step=custom_exc,
        )
        ds = build_transitions([log])
        np.testing.assert_array_equal(ds.u, custom_exc[:-1])

    def test_state_dim_consistency_check(self) -> None:
        a = _make_log(n_steps=5, qpos_dim=2)
        b = _make_log(n_steps=5, qpos_dim=3, episode_id=1)
        with pytest.raises(ValueError, match="state_dim mismatch"):
            build_transitions([a, b])

    def test_dx_equals_x_next_minus_x(self) -> None:
        log = _make_log(episode_id=0, n_steps=4)
        ds = build_transitions([log])
        np.testing.assert_allclose(ds.dx, ds.x_next - ds.x, atol=1e-5)

    def test_metadata_keys_present(self) -> None:
        log = _make_log(
            episode_id=42,
            n_steps=3,
            target_id="train:42",
            target_seed=42,
            sdn_sigma=0.1,
            obs_noise_sigma={"qpos": 0.05},
            obs_delay_steps=2,
        )
        ds = build_transitions([log])
        meta = ds.episode_metadata[0]
        assert meta["episode_id"] == 42
        assert meta["target_id"] == "train:42"
        assert meta["target_seed"] == 42
        assert meta["sdn_sigma"] == pytest.approx(0.1)
        assert meta["obs_noise_sigma"] == {"qpos": 0.05}
        assert meta["obs_delay_steps"] == 2
        assert meta["n_steps"] == 3
        assert meta["transitions_used"] == 2

    def test_non_episode_log_input_raises(self) -> None:
        with pytest.raises(ValueError, match="EpisodeLog"):
            build_transitions(["not a log"])  # type: ignore[list-item]


# --- shuffle_transitions ---


class TestShuffleTransitions:
    def test_same_seed_same_permutation(self) -> None:
        ds = build_transitions([_make_log(n_steps=10, episode_id=i) for i in range(3)])
        a = shuffle_transitions(ds, rng=42)
        b = shuffle_transitions(ds, rng=42)
        np.testing.assert_array_equal(a.x, b.x)
        np.testing.assert_array_equal(a.episode_index, b.episode_index)

    def test_different_seed_different_permutation(self) -> None:
        ds = build_transitions([_make_log(n_steps=10, episode_id=i) for i in range(3)])
        a = shuffle_transitions(ds, rng=1)
        b = shuffle_transitions(ds, rng=2)
        # Expect at least one row to differ.
        assert not np.array_equal(a.x, b.x)

    def test_shuffle_preserves_membership(self) -> None:
        ds = build_transitions([_make_log(n_steps=5, episode_id=i) for i in range(3)])
        shuffled = shuffle_transitions(ds, rng=0)
        # Sorted rows must match (ignoring order).
        np.testing.assert_array_equal(
            np.sort(shuffled.x.flatten()),
            np.sort(ds.x.flatten()),
        )
        # episode_index value counts must match.
        original_counts = np.bincount(ds.episode_index, minlength=ds.n_episodes)
        shuffled_counts = np.bincount(
            shuffled.episode_index, minlength=ds.n_episodes
        )
        np.testing.assert_array_equal(original_counts, shuffled_counts)

    def test_metadata_unchanged(self) -> None:
        ds = build_transitions([_make_log(n_steps=5, episode_id=i) for i in range(2)])
        shuffled = shuffle_transitions(ds, rng=0)
        assert shuffled.episode_metadata == ds.episode_metadata
        assert shuffled.n_episodes == ds.n_episodes

    def test_bool_rng_raises(self) -> None:
        ds = _good_dataset()
        with pytest.raises(TypeError):
            shuffle_transitions(ds, rng=True)  # type: ignore[arg-type]

    def test_str_rng_raises(self) -> None:
        ds = _good_dataset()
        with pytest.raises(TypeError):
            shuffle_transitions(ds, rng="42")  # type: ignore[arg-type]

    def test_generator_input(self) -> None:
        ds = build_transitions([_make_log(n_steps=5)])
        gen = np.random.default_rng(0)
        shuffled = shuffle_transitions(ds, rng=gen)
        assert shuffled.n == ds.n


# --- split_by_episode ---


class TestSplitByEpisode:
    def test_basic_split(self) -> None:
        # Episodes with id 10, 11, 12 each contributing 4 transitions.
        logs = [_make_log(episode_id=10 + i, n_steps=5) for i in range(3)]
        ds = build_transitions(logs)
        train, val = split_by_episode(ds, val_episode_ids=[11])
        assert val.n == 4
        assert train.n == 8
        assert val.n_episodes == 1
        assert train.n_episodes == 2
        # Original IDs preserved in metadata.
        assert val.episode_metadata[0]["episode_id"] == 11
        train_ids = sorted(m["episode_id"] for m in train.episode_metadata)
        assert train_ids == [10, 12]

    def test_episode_index_reindexed(self) -> None:
        logs = [_make_log(episode_id=10 + i, n_steps=5) for i in range(3)]
        ds = build_transitions(logs)
        train, val = split_by_episode(ds, val_episode_ids=[11])
        # train has 2 episodes → episode_index in [0, 2)
        assert int(train.episode_index.min()) == 0
        assert int(train.episode_index.max()) == 1
        # val has 1 episode → episode_index all 0
        np.testing.assert_array_equal(
            val.episode_index, np.zeros(val.n, dtype=np.int64)
        )

    def test_unknown_id_raises(self) -> None:
        logs = [_make_log(episode_id=10 + i, n_steps=5) for i in range(3)]
        ds = build_transitions(logs)
        with pytest.raises(ValueError, match="not in dataset"):
            split_by_episode(ds, val_episode_ids=[999])

    def test_empty_val(self) -> None:
        logs = [_make_log(episode_id=i, n_steps=5) for i in range(2)]
        ds = build_transitions(logs)
        train, val = split_by_episode(ds, val_episode_ids=[])
        assert val.n == 0
        assert val.n_episodes == 0
        assert train.n == ds.n
        assert train.n_episodes == ds.n_episodes

    def test_all_to_val(self) -> None:
        logs = [_make_log(episode_id=i, n_steps=5) for i in range(2)]
        ds = build_transitions(logs)
        train, val = split_by_episode(ds, val_episode_ids=[0, 1])
        assert val.n == ds.n
        assert train.n == 0
        assert train.n_episodes == 0

    def test_save_load_round_trip_after_split(self, tmp_path: Path) -> None:
        logs = [_make_log(episode_id=i, n_steps=5) for i in range(3)]
        ds = build_transitions(logs)
        train, _val = split_by_episode(ds, val_episode_ids=[1])
        path = tmp_path / "train.npz"
        train.save(path)
        loaded = TransitionDataset.load(path)
        np.testing.assert_array_equal(train.x, loaded.x)
        assert loaded.episode_metadata == train.episode_metadata
