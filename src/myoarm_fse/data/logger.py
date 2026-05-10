"""Run-level helpers: ``run_id``, ``index.json`` read/write.

Each collection produces a directory ``runs/episodes/{run_id}/`` with:

```text
index.json     # run metadata + per-episode entries
0000.npz       # episode 0 (saved via EpisodeLog.save)
0001.npz
...
```

The index is the single source of truth for "what episodes exist in this
run, and where". ``EpisodeLog.save`` writes the per-episode npz; this
module only handles the index.

``run_id`` is a UTC timestamp formatted to be filesystem-safe
(``2026-05-10T08-30-15Z``) — sortable, unambiguous, no escaping needed.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# Filesystem-safe ISO 8601 (no colons): YYYY-MM-DDTHH-MM-SSZ.
_RUN_ID_FORMAT: str = "%Y-%m-%dT%H-%M-%SZ"


def make_run_id(now: datetime | None = None) -> str:
    """Return a filesystem-safe UTC timestamp suitable as a run id."""
    t = now if now is not None else datetime.now(timezone.utc)
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    return t.astimezone(timezone.utc).strftime(_RUN_ID_FORMAT)


def hash_config(config: dict[str, Any]) -> str:
    """Deterministic short hash of a JSON-serializable config dict."""
    payload = json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]


@dataclass(frozen=True)
class IndexEntry:
    """One row of ``index.json`` describing a saved episode file."""

    episode_id: int
    file: str
    target_id: str
    target_seed: int
    n_steps: int


@dataclass
class RunIndex:
    """In-memory representation of ``index.json``."""

    run_id: str
    created_at: str
    config_hash: str
    config: dict[str, Any]
    target_set_path: str
    episodes: list[IndexEntry] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "created_at": self.created_at,
            "config_hash": self.config_hash,
            "config": self.config,
            "target_set_path": self.target_set_path,
            "episodes": [asdict(e) for e in self.episodes],
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> RunIndex:
        with open(path) as f:
            data = json.load(f)
        return cls(
            run_id=data["run_id"],
            created_at=data["created_at"],
            config_hash=data["config_hash"],
            config=data["config"],
            target_set_path=data["target_set_path"],
            episodes=[IndexEntry(**e) for e in data["episodes"]],
        )

    def append(self, entry: IndexEntry) -> None:
        if any(e.episode_id == entry.episode_id for e in self.episodes):
            raise ValueError(
                f"episode_id {entry.episode_id} already in index"
            )
        self.episodes.append(entry)
