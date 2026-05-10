"""Tests for myoarm_fse.data.schema.EpisodeLog."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from myoarm_fse.data import EpisodeLog


def _make_log(
    *,
    n_steps: int = 4,
    qpos_dim: int = 2,
    qvel_dim: int = 3,
    act_dim: int = 5,
    action_dim: int = 5,
    max_steps: int = 10,
) -> EpisodeLog:
    T = n_steps
    f32 = np.float32

    def cart() -> np.ndarray:
        return np.zeros((T, 3), dtype=f32)

    return EpisodeLog(
        episode_id=0,
        target_id="train:0",
        target_split="train",
        target_seed=0,
        target_pos_set=np.array([0.1, 0.2, 0.3], dtype=f32),
        controller_name="random",
        controller_seed=1,
        sdn_sigma=0.0,
        sdn_seed=2,
        obs_noise_sigma={},
        obs_noise_seed=3,
        obs_delay_steps=0,
        obs_compose="noisy_then_delayed",
        max_steps=max_steps,
        n_steps=T,
        created_at="2026-05-10T08:30:15Z",
        config_hash="abcd1234",
        step=np.arange(T, dtype=np.int64),
        time=np.arange(T, dtype=f32) * 0.02,
        true_qpos=np.zeros((T, qpos_dim), dtype=f32),
        true_qvel=np.zeros((T, qvel_dim), dtype=f32),
        true_act=np.zeros((T, act_dim), dtype=f32),
        true_tip_pos=cart(),
        true_target_pos=cart(),
        true_reach_err=cart(),
        obs_qpos=np.zeros((T, qpos_dim), dtype=f32),
        obs_qvel=np.zeros((T, qvel_dim), dtype=f32),
        obs_act=np.zeros((T, act_dim), dtype=f32),
        obs_tip_pos=cart(),
        obs_target_pos=cart(),
        obs_reach_err=cart(),
        neural_command=np.zeros((T, action_dim), dtype=f32),
        excitation_command=np.zeros((T, action_dim), dtype=f32),
        excitation=np.zeros((T, action_dim), dtype=f32),
        api_action=np.zeros((T, action_dim), dtype=f32),
        last_ctrl=np.zeros((T, action_dim), dtype=f32),
        reward=np.zeros(T, dtype=f32),
        terminated=np.zeros(T, dtype=np.bool_),
        truncated=np.zeros(T, dtype=np.bool_),
        meta={"note": "synthetic"},
    )


# --- construction validation ---


class TestConstruction:
    def test_basic(self) -> None:
        log = _make_log()
        assert log.n_steps == 4
        assert log.action_dim == 5

    def test_n_steps_zero_allowed(self) -> None:
        log = _make_log(n_steps=0)
        assert log.n_steps == 0

    def test_negative_n_steps_raises(self) -> None:
        with pytest.raises(ValueError):
            _make_log(n_steps=-1)

    def test_max_steps_below_n_steps_raises(self) -> None:
        with pytest.raises(ValueError, match="max_steps"):
            _make_log(n_steps=10, max_steps=5)

    def test_target_pos_set_shape(self) -> None:
        log = _make_log()
        bad = np.array([0.0, 0.1], dtype=np.float32)
        with pytest.raises(ValueError, match=r"shape \(3,\)"):
            EpisodeLog(**{**_log_kwargs(log), "target_pos_set": bad})

    def test_target_pos_set_dtype(self) -> None:
        log = _make_log()
        bad = np.array([0.0, 0.1, 0.2], dtype=np.float64)
        with pytest.raises(ValueError, match="float32"):
            EpisodeLog(**{**_log_kwargs(log), "target_pos_set": bad})

    def test_step_array_wrong_dtype(self) -> None:
        log = _make_log()
        bad_step = np.arange(log.n_steps, dtype=np.int32)
        with pytest.raises(ValueError, match="int64"):
            EpisodeLog(**{**_log_kwargs(log), "step": bad_step})

    def test_step_array_length_mismatch(self) -> None:
        log = _make_log()
        # n_steps=4 but reward only length 3
        with pytest.raises(ValueError, match="must equal n_steps"):
            EpisodeLog(
                **{**_log_kwargs(log), "reward": np.zeros(3, dtype=np.float32)}
            )

    def test_cart_inner_shape_mismatch(self) -> None:
        log = _make_log()
        bad = np.zeros((log.n_steps, 4), dtype=np.float32)
        with pytest.raises(ValueError, match=r"shape\[1:\]"):
            EpisodeLog(**{**_log_kwargs(log), "true_tip_pos": bad})

    def test_meta_must_be_json_serializable(self) -> None:
        log = _make_log()
        with pytest.raises(ValueError, match="JSON"):
            EpisodeLog(**{**_log_kwargs(log), "meta": {"x": object()}})

    def test_obs_noise_sigma_not_json(self) -> None:
        log = _make_log()
        bad = {"x": object()}
        with pytest.raises(ValueError, match="JSON"):
            EpisodeLog(**{**_log_kwargs(log), "obs_noise_sigma": bad})


# --- save / load ---


class TestSaveLoad:
    def test_round_trip(self, tmp_path: Path) -> None:
        original = _make_log(n_steps=4, action_dim=5)
        path = tmp_path / "ep.npz"
        original.save(path)
        loaded = EpisodeLog.load(path)
        for name in (
            "episode_id",
            "target_id",
            "target_split",
            "target_seed",
            "controller_name",
            "controller_seed",
            "sdn_sigma",
            "sdn_seed",
            "obs_noise_seed",
            "obs_delay_steps",
            "obs_compose",
            "max_steps",
            "n_steps",
            "created_at",
            "config_hash",
        ):
            assert getattr(loaded, name) == getattr(original, name), name
        np.testing.assert_array_equal(loaded.target_pos_set, original.target_pos_set)
        np.testing.assert_array_equal(loaded.step, original.step)
        np.testing.assert_array_equal(loaded.api_action, original.api_action)
        np.testing.assert_array_equal(loaded.terminated, original.terminated)
        assert loaded.obs_noise_sigma == original.obs_noise_sigma
        assert loaded.meta == original.meta

    def test_save_creates_parent_dir(self, tmp_path: Path) -> None:
        log = _make_log()
        path = tmp_path / "deep" / "subdir" / "ep.npz"
        log.save(path)
        assert path.exists()

    def test_load_disallows_pickle(self, tmp_path: Path) -> None:
        path = tmp_path / "ep.npz"
        _make_log().save(path)
        with np.load(path, allow_pickle=False) as f:
            assert "meta_json" in f
            assert "step" in f


# --- helpers ---


def _log_kwargs(log: EpisodeLog) -> dict[str, Any]:
    """Extract the constructor kwargs from a log so tests can override one field."""
    from dataclasses import fields

    return {f.name: getattr(log, f.name) for f in fields(log)}


def test_action_dim_property() -> None:
    log = _make_log(action_dim=7)
    assert log.action_dim == 7


def test_action_dim_zero_steps() -> None:
    log = _make_log(n_steps=0, action_dim=5)
    # Even with zero steps, api_action.shape[1] is preserved.
    assert log.action_dim == 5
