"""Tests for myoarm_fse.data.logger (run id, index.json)."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

import pytest

from myoarm_fse.data import IndexEntry, RunIndex, hash_config, make_run_id


# --- run_id ---


def test_run_id_format_is_filesystem_safe() -> None:
    rid = make_run_id(datetime(2026, 5, 10, 8, 30, 15, tzinfo=timezone.utc))
    assert rid == "2026-05-10T08-30-15Z"


def test_run_id_default_uses_utc_now() -> None:
    rid = make_run_id()
    # Match shape only.
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}Z", rid)


def test_run_id_naive_datetime_assumed_utc() -> None:
    naive = datetime(2026, 5, 10, 0, 0, 0)
    rid = make_run_id(naive)
    assert rid == "2026-05-10T00-00-00Z"


def test_run_id_non_utc_timezone_normalized() -> None:
    from datetime import timedelta

    jst = timezone(timedelta(hours=9))
    t = datetime(2026, 5, 10, 17, 30, 15, tzinfo=jst)
    rid = make_run_id(t)
    # 17:30 JST = 08:30 UTC.
    assert rid == "2026-05-10T08-30-15Z"


# --- hash_config ---


def test_hash_config_deterministic() -> None:
    a = hash_config({"x": 1, "y": [1, 2, 3]})
    b = hash_config({"y": [1, 2, 3], "x": 1})
    assert a == b
    assert len(a) == 12


def test_hash_config_changes_with_content() -> None:
    a = hash_config({"x": 1})
    b = hash_config({"x": 2})
    assert a != b


# --- RunIndex ---


def _entry(eid: int) -> IndexEntry:
    return IndexEntry(
        episode_id=eid,
        file=f"{eid:04d}.npz",
        target_id=f"train:{eid}",
        target_seed=eid,
        n_steps=100,
    )


def test_index_round_trip(tmp_path: Path) -> None:
    idx = RunIndex(
        run_id="2026-05-10T08-30-15Z",
        created_at="2026-05-10T08:30:15Z",
        config_hash="abcd1234",
        config={"env_id": "myoArmReachFixed-v0"},
        target_set_path="runs/targets/train.npz",
        episodes=[_entry(0), _entry(1)],
    )
    path = tmp_path / "index.json"
    idx.save(path)
    loaded = RunIndex.load(path)
    assert loaded.run_id == idx.run_id
    assert loaded.config == idx.config
    assert len(loaded.episodes) == 2
    assert loaded.episodes[0].episode_id == 0
    assert loaded.episodes[1].file == "0001.npz"


def test_index_append_rejects_duplicate() -> None:
    idx = RunIndex(
        run_id="x",
        created_at="x",
        config_hash="x",
        config={},
        target_set_path="",
    )
    idx.append(_entry(0))
    with pytest.raises(ValueError, match="already"):
        idx.append(_entry(0))


def test_index_save_creates_parent(tmp_path: Path) -> None:
    idx = RunIndex(
        run_id="x", created_at="y", config_hash="z",
        config={}, target_set_path="",
    )
    path = tmp_path / "deep" / "subdir" / "index.json"
    idx.save(path)
    assert path.exists()
