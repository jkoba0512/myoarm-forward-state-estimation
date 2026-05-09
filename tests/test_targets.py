"""Pure unit tests for myoarm_fse.envs.targets (no MyoSuite required)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from myoarm_fse.envs.targets import (
    SplitConfig,
    TargetGenerationConfig,
    TargetSet,
    generate_seed_list,
)


# --- generate_seed_list ---


class TestSeedList:
    def test_basic_rule(self) -> None:
        seeds = generate_seed_list(generator_seed=10, seed_offset=100, n=5)
        np.testing.assert_array_equal(seeds, [110, 111, 112, 113, 114])
        assert seeds.dtype == np.int64

    def test_zero_offset(self) -> None:
        seeds = generate_seed_list(generator_seed=0, seed_offset=0, n=3)
        np.testing.assert_array_equal(seeds, [0, 1, 2])

    @pytest.mark.parametrize("bad", [0, -1])
    def test_non_positive_n_raises(self, bad: int) -> None:
        with pytest.raises(ValueError):
            generate_seed_list(generator_seed=0, seed_offset=0, n=bad)

    def test_bool_n_raises(self) -> None:
        with pytest.raises(ValueError):
            generate_seed_list(generator_seed=0, seed_offset=0, n=True)  # type: ignore[arg-type]

    def test_negative_offset_raises(self) -> None:
        with pytest.raises(ValueError):
            generate_seed_list(generator_seed=0, seed_offset=-1, n=3)

    def test_negative_generator_seed_raises(self) -> None:
        with pytest.raises(ValueError):
            generate_seed_list(generator_seed=-1, seed_offset=0, n=3)


# --- SplitConfig ---


class TestSplitConfig:
    def test_valid(self) -> None:
        s = SplitConfig(name="train", n=200, seed_offset=0)
        assert s.name == "train"
        assert s.n == 200
        assert s.seed_offset == 0

    def test_empty_name_raises(self) -> None:
        with pytest.raises(ValueError):
            SplitConfig(name="", n=1, seed_offset=0)

    def test_zero_n_raises(self) -> None:
        with pytest.raises(ValueError):
            SplitConfig(name="x", n=0, seed_offset=0)

    def test_negative_seed_offset_raises(self) -> None:
        with pytest.raises(ValueError):
            SplitConfig(name="x", n=1, seed_offset=-1)

    def test_bool_n_raises(self) -> None:
        with pytest.raises(ValueError):
            SplitConfig(name="x", n=True, seed_offset=0)  # type: ignore[arg-type]


# --- TargetGenerationConfig ---


def _good_config_dict() -> dict:
    return {
        "env_id": "myoArmReachRandom-v0",
        "generator_seed": 0,
        "output_dir": "runs/targets",
        "splits": {
            "train": {"n": 10, "seed_offset": 0},
            "val": {"n": 5, "seed_offset": 1000},
        },
    }


class TestConfigFromDict:
    def test_valid(self) -> None:
        c = TargetGenerationConfig.from_dict(_good_config_dict())
        assert c.env_id == "myoArmReachRandom-v0"
        assert c.generator_seed == 0
        assert c.output_dir == "runs/targets"
        assert len(c.splits) == 2
        assert {s.name for s in c.splits} == {"train", "val"}

    def test_split_lookup(self) -> None:
        c = TargetGenerationConfig.from_dict(_good_config_dict())
        assert c.split("train").n == 10
        with pytest.raises(KeyError):
            c.split("missing")

    def test_missing_required_key(self) -> None:
        d = _good_config_dict()
        del d["env_id"]
        with pytest.raises(ValueError, match="missing required keys"):
            TargetGenerationConfig.from_dict(d)

    def test_unknown_top_level_key(self) -> None:
        d = _good_config_dict()
        d["banana"] = 1
        with pytest.raises(ValueError, match="unknown keys"):
            TargetGenerationConfig.from_dict(d)

    def test_unknown_split_key(self) -> None:
        d = _good_config_dict()
        d["splits"]["train"]["banana"] = 1
        with pytest.raises(ValueError, match="unknown keys"):
            TargetGenerationConfig.from_dict(d)

    def test_missing_split_key(self) -> None:
        d = _good_config_dict()
        del d["splits"]["train"]["seed_offset"]
        with pytest.raises(ValueError, match="missing keys"):
            TargetGenerationConfig.from_dict(d)

    def test_empty_splits_raises(self) -> None:
        d = _good_config_dict()
        d["splits"] = {}
        with pytest.raises(ValueError, match="at least one"):
            TargetGenerationConfig.from_dict(d)

    def test_overlapping_seed_ranges_raises(self) -> None:
        d = _good_config_dict()
        # Overlap: train [0, 10), val [5, 10) — overlaps in [5, 10).
        d["splits"]["val"]["seed_offset"] = 5
        d["splits"]["val"]["n"] = 5
        with pytest.raises(ValueError, match="overlap"):
            TargetGenerationConfig.from_dict(d)

    def test_negative_generator_seed_raises(self) -> None:
        d = _good_config_dict()
        d["generator_seed"] = -1
        with pytest.raises(ValueError):
            TargetGenerationConfig.from_dict(d)


class TestConfigFromYAML:
    def test_round_trip(self, tmp_path: Path) -> None:
        path = tmp_path / "cfg.yaml"
        with open(path, "w") as f:
            yaml.safe_dump(_good_config_dict(), f)
        c = TargetGenerationConfig.from_yaml(path)
        assert c.env_id == "myoArmReachRandom-v0"
        assert c.split("val").seed_offset == 1000


# --- TargetSet ---


def _good_target_set(n: int = 3, split: str = "train") -> TargetSet:
    seeds = np.arange(n, dtype=np.int64)
    target_pos = np.linspace(0.1, 1.0, n * 3, dtype=np.float32).reshape(n, 3)
    init_dist = np.linspace(0.1, 0.5, n, dtype=np.float32)
    return TargetSet(
        split=split,
        seeds=seeds,
        target_pos=target_pos,
        tip_to_target_init_distance=init_dist,
        meta={"env_id": "myoArmReachRandom-v0", "n": n},
    )


class TestTargetSetConstruction:
    def test_valid(self) -> None:
        ts = _good_target_set()
        assert ts.n == 3
        assert ts.split == "train"

    def test_seeds_must_be_ndarray(self) -> None:
        with pytest.raises(ValueError, match="np.ndarray"):
            TargetSet(
                split="x",
                seeds=[0, 1, 2],  # type: ignore[arg-type]
                target_pos=np.zeros((3, 3), dtype=np.float32),
                tip_to_target_init_distance=np.zeros(3, dtype=np.float32),
            )

    def test_seeds_wrong_dtype(self) -> None:
        with pytest.raises(ValueError, match="dtype"):
            TargetSet(
                split="x",
                seeds=np.zeros(3, dtype=np.int32),
                target_pos=np.zeros((3, 3), dtype=np.float32),
                tip_to_target_init_distance=np.zeros(3, dtype=np.float32),
            )

    def test_duplicate_seeds_raises(self) -> None:
        with pytest.raises(ValueError, match="duplicate"):
            TargetSet(
                split="x",
                seeds=np.array([0, 1, 1], dtype=np.int64),
                target_pos=np.zeros((3, 3), dtype=np.float32),
                tip_to_target_init_distance=np.zeros(3, dtype=np.float32),
            )

    def test_target_pos_wrong_shape(self) -> None:
        with pytest.raises(ValueError, match=r"shape \(3, 3\)"):
            TargetSet(
                split="x",
                seeds=np.arange(3, dtype=np.int64),
                target_pos=np.zeros((3, 4), dtype=np.float32),
                tip_to_target_init_distance=np.zeros(3, dtype=np.float32),
            )

    def test_target_pos_wrong_dtype(self) -> None:
        with pytest.raises(ValueError, match="dtype"):
            TargetSet(
                split="x",
                seeds=np.arange(3, dtype=np.int64),
                target_pos=np.zeros((3, 3), dtype=np.float64),
                tip_to_target_init_distance=np.zeros(3, dtype=np.float32),
            )

    def test_target_pos_nan_raises(self) -> None:
        bad = np.zeros((3, 3), dtype=np.float32)
        bad[0, 0] = np.nan
        with pytest.raises(ValueError, match="non-finite"):
            TargetSet(
                split="x",
                seeds=np.arange(3, dtype=np.int64),
                target_pos=bad,
                tip_to_target_init_distance=np.zeros(3, dtype=np.float32),
            )

    def test_init_dist_wrong_shape(self) -> None:
        with pytest.raises(ValueError, match=r"shape \(3,\)"):
            TargetSet(
                split="x",
                seeds=np.arange(3, dtype=np.int64),
                target_pos=np.zeros((3, 3), dtype=np.float32),
                tip_to_target_init_distance=np.zeros(4, dtype=np.float32),
            )

    def test_init_dist_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            TargetSet(
                split="x",
                seeds=np.arange(3, dtype=np.int64),
                target_pos=np.zeros((3, 3), dtype=np.float32),
                tip_to_target_init_distance=np.array([0.1, -0.1, 0.2], dtype=np.float32),
            )

    def test_meta_must_be_dict(self) -> None:
        with pytest.raises(ValueError, match="dict"):
            TargetSet(
                split="x",
                seeds=np.arange(3, dtype=np.int64),
                target_pos=np.zeros((3, 3), dtype=np.float32),
                tip_to_target_init_distance=np.zeros(3, dtype=np.float32),
                meta="not a dict",  # type: ignore[arg-type]
            )

    def test_meta_must_be_json_serializable(self) -> None:
        with pytest.raises(ValueError, match="JSON-serializable"):
            TargetSet(
                split="x",
                seeds=np.arange(3, dtype=np.int64),
                target_pos=np.zeros((3, 3), dtype=np.float32),
                tip_to_target_init_distance=np.zeros(3, dtype=np.float32),
                meta={"bad": object()},
            )

    def test_empty_split_name_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            TargetSet(
                split="",
                seeds=np.arange(3, dtype=np.int64),
                target_pos=np.zeros((3, 3), dtype=np.float32),
                tip_to_target_init_distance=np.zeros(3, dtype=np.float32),
            )


class TestSaveLoad:
    def test_round_trip(self, tmp_path: Path) -> None:
        original = _good_target_set(n=5, split="val")
        path = tmp_path / "val.npz"
        original.save(path)
        loaded = TargetSet.load(path)
        assert loaded.split == original.split
        np.testing.assert_array_equal(loaded.seeds, original.seeds)
        np.testing.assert_array_equal(loaded.target_pos, original.target_pos)
        np.testing.assert_array_equal(
            loaded.tip_to_target_init_distance,
            original.tip_to_target_init_distance,
        )
        assert loaded.meta == original.meta

    def test_save_creates_parent_dir(self, tmp_path: Path) -> None:
        path = tmp_path / "deep" / "subdir" / "train.npz"
        ts = _good_target_set()
        ts.save(path)
        assert path.exists()

    def test_load_disallows_pickle(self, tmp_path: Path) -> None:
        # Confirm that load uses allow_pickle=False (pickle metadata would
        # raise on load). We verify by loading a file we just saved with
        # safe primitives — it should succeed; then confirm the npz does
        # not contain pickle markers in our serialization.
        path = tmp_path / "x.npz"
        _good_target_set().save(path)
        with np.load(path, allow_pickle=False) as f:
            # Just touching the keys should work without errors.
            assert "seeds" in f
            assert "meta_json" in f


class TestRoundTripValues:
    def test_meta_json_preserved(self, tmp_path: Path) -> None:
        ts = TargetSet(
            split="train",
            seeds=np.arange(2, dtype=np.int64),
            target_pos=np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32),
            tip_to_target_init_distance=np.array([0.7, 0.8], dtype=np.float32),
            meta={
                "env_id": "myoArmReachRandom-v0",
                "nested": {"a": 1, "b": [1, 2, 3]},
                "site_low": [-0.35, -0.42, 0.98],
                "site_high": [0.0, -0.07, 1.83],
                "workspace_bounds": None,
            },
        )
        path = tmp_path / "x.npz"
        ts.save(path)
        loaded = TargetSet.load(path)
        assert loaded.meta == ts.meta
        # Confirm meta is round-trippable as JSON
        assert json.loads(json.dumps(loaded.meta)) == ts.meta
