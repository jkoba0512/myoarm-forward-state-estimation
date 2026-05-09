"""Target set generator for myoArm reaching.

Builds reproducible train / val / test / extrapolation splits of target
positions for ``myoArmReachRandom-v0``.

Probe-driven design (see Step 1 completion log):

- ``env.reset(seed=k)`` does NOT reproduce target_pos. MyoSuite's
  ``ReachEnvV0.reset`` calls ``generate_target_pose()`` *before*
  ``super().reset(seed=...)``, so the seed argument never controls the
  target draw of that reset call. Even ``reset(seed=k); reset(seed=k)``
  still produces different targets.
- The reliable path: write a target directly to
  ``env.unwrapped.mj_model.site_pos[<target_site_id>]`` and call
  ``mujoco.mj_forward(...)``. ``mj_data.site_xpos`` then reflects the
  written value, and the value persists across ``step()``.

This module therefore owns the RNG: each target is drawn from
``np.random.default_rng(seed_i).uniform(low, high)`` where
``(low, high)`` come from the env's intrinsic ``target_reach_range``.
The env's ``np_random`` is never relied on for reproducibility.

Seed rule (per Step 1 design decisions):

```text
seed_i = generator_seed + seed_offset + i
```

MyoSuite is imported lazily inside ``generate_*`` functions so that
``import myoarm_fse.envs.targets`` does not pay the registration cost.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

GENERATOR_VERSION: str = "v0"

_DTYPE_TARGET: np.dtype = np.dtype(np.float32)
_DTYPE_DIST: np.dtype = np.dtype(np.float32)
_DTYPE_SEED: np.dtype = np.dtype(np.int64)
_CART_DIM: int = 3
_VALID_SPLIT_KEYS: tuple[str, ...] = ("n", "seed_offset")


# --- helpers ---


def _check_positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive int, got bool: {value!r}")
    if not isinstance(value, int):
        raise ValueError(
            f"{name} must be a positive int, "
            f"got {type(value).__name__}: {value!r}"
        )
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")
    return value


def _check_non_negative_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a non-negative int, got bool: {value!r}")
    if not isinstance(value, int):
        raise ValueError(
            f"{name} must be a non-negative int, "
            f"got {type(value).__name__}: {value!r}"
        )
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")
    return value


# --- config ---


@dataclass(frozen=True)
class SplitConfig:
    """Per-split parameters."""

    name: str
    n: int
    seed_offset: int

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError(f"split name must be a non-empty str, got {self.name!r}")
        _check_positive_int(self.n, f"splits[{self.name!r}].n")
        _check_non_negative_int(self.seed_offset, f"splits[{self.name!r}].seed_offset")


@dataclass(frozen=True)
class TargetGenerationConfig:
    """Top-level config for target set generation."""

    env_id: str
    generator_seed: int
    output_dir: str
    splits: tuple[SplitConfig, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.env_id, str) or not self.env_id:
            raise ValueError(f"env_id must be a non-empty str, got {self.env_id!r}")
        _check_non_negative_int(self.generator_seed, "generator_seed")
        if not isinstance(self.output_dir, str) or not self.output_dir:
            raise ValueError(
                f"output_dir must be a non-empty str, got {self.output_dir!r}"
            )
        if not self.splits:
            raise ValueError("splits must contain at least one entry")
        names = [s.name for s in self.splits]
        if len(set(names)) != len(names):
            raise ValueError(f"split names must be unique, got {names}")
        # Detect global seed collisions across splits.
        all_seeds: list[int] = []
        for s in self.splits:
            all_seeds.extend(
                range(
                    self.generator_seed + s.seed_offset,
                    self.generator_seed + s.seed_offset + s.n,
                )
            )
        if len(set(all_seeds)) != len(all_seeds):
            raise ValueError(
                "split seed ranges overlap; adjust seed_offset or n so the "
                "(generator_seed + seed_offset + i) ranges are disjoint"
            )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TargetGenerationConfig:
        if not isinstance(data, dict):
            raise ValueError(
                f"config must be a mapping, got {type(data).__name__}"
            )
        required = {"env_id", "generator_seed", "output_dir", "splits"}
        missing = required - data.keys()
        if missing:
            raise ValueError(f"config missing required keys: {sorted(missing)}")
        unknown = data.keys() - required
        if unknown:
            raise ValueError(f"config has unknown keys: {sorted(unknown)}")
        splits_data = data["splits"]
        if not isinstance(splits_data, dict):
            raise ValueError(
                f"splits must be a mapping, got {type(splits_data).__name__}"
            )
        splits: list[SplitConfig] = []
        for name, body in splits_data.items():
            if not isinstance(body, dict):
                raise ValueError(
                    f"splits[{name!r}] must be a mapping, "
                    f"got {type(body).__name__}"
                )
            unknown_keys = body.keys() - set(_VALID_SPLIT_KEYS)
            if unknown_keys:
                raise ValueError(
                    f"splits[{name!r}] has unknown keys: {sorted(unknown_keys)}"
                )
            missing_keys = set(_VALID_SPLIT_KEYS) - body.keys()
            if missing_keys:
                raise ValueError(
                    f"splits[{name!r}] missing keys: {sorted(missing_keys)}"
                )
            splits.append(
                SplitConfig(name=name, n=body["n"], seed_offset=body["seed_offset"])
            )
        return cls(
            env_id=data["env_id"],
            generator_seed=data["generator_seed"],
            output_dir=data["output_dir"],
            splits=tuple(splits),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> TargetGenerationConfig:
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    def split(self, name: str) -> SplitConfig:
        for s in self.splits:
            if s.name == name:
                return s
        raise KeyError(f"unknown split {name!r}; have {[s.name for s in self.splits]}")


# --- TargetSet ---


@dataclass(frozen=True, eq=False)
class TargetSet:
    """A reproducible split of target positions for myoArm reaching."""

    split: str
    seeds: np.ndarray
    target_pos: np.ndarray
    tip_to_target_init_distance: np.ndarray
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.split, str) or not self.split:
            raise ValueError(f"split must be a non-empty str, got {self.split!r}")
        # seeds
        if not isinstance(self.seeds, np.ndarray):
            raise ValueError(
                f"seeds must be np.ndarray, got {type(self.seeds).__name__}"
            )
        if self.seeds.ndim != 1:
            raise ValueError(f"seeds must be 1-D, got shape={self.seeds.shape}")
        if not np.issubdtype(self.seeds.dtype, np.integer):
            raise ValueError(f"seeds must be integer dtype, got {self.seeds.dtype}")
        if self.seeds.dtype != _DTYPE_SEED:
            raise ValueError(
                f"seeds must have dtype {_DTYPE_SEED}, got {self.seeds.dtype}"
            )
        if len(np.unique(self.seeds)) != len(self.seeds):
            raise ValueError("seeds contain duplicates")
        n = self.seeds.shape[0]
        # target_pos
        if not isinstance(self.target_pos, np.ndarray):
            raise ValueError("target_pos must be np.ndarray")
        if self.target_pos.shape != (n, _CART_DIM):
            raise ValueError(
                f"target_pos must have shape ({n}, 3), got {self.target_pos.shape}"
            )
        if self.target_pos.dtype != _DTYPE_TARGET:
            raise ValueError(
                f"target_pos must have dtype {_DTYPE_TARGET}, "
                f"got {self.target_pos.dtype}"
            )
        if not np.isfinite(self.target_pos).all():
            raise ValueError("target_pos contains non-finite values")
        # tip_to_target_init_distance
        d = self.tip_to_target_init_distance
        if not isinstance(d, np.ndarray):
            raise ValueError("tip_to_target_init_distance must be np.ndarray")
        if d.shape != (n,):
            raise ValueError(
                f"tip_to_target_init_distance must have shape ({n},), got {d.shape}"
            )
        if d.dtype != _DTYPE_DIST:
            raise ValueError(
                f"tip_to_target_init_distance must have dtype {_DTYPE_DIST}, "
                f"got {d.dtype}"
            )
        if not np.isfinite(d).all():
            raise ValueError(
                "tip_to_target_init_distance contains non-finite values"
            )
        if (d < 0.0).any():
            raise ValueError("tip_to_target_init_distance must be non-negative")
        # meta
        if not isinstance(self.meta, dict):
            raise ValueError("meta must be a dict")
        # Verify meta is JSON-serializable now (fail fast at construction).
        try:
            json.dumps(self.meta)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"meta must be JSON-serializable: {exc}") from exc

    @property
    def n(self) -> int:
        return int(self.seeds.shape[0])

    def save(self, path: str | Path) -> None:
        """Save to ``path`` as ``.npz`` (no pickle, JSON-encoded meta)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta_json = json.dumps(self.meta)
        # np.savez auto-appends .npz unless caller already used it.
        np.savez(
            path,
            split=np.array(self.split),
            seeds=self.seeds,
            target_pos=self.target_pos,
            tip_to_target_init_distance=self.tip_to_target_init_distance,
            meta_json=np.array(meta_json),
        )

    @classmethod
    def load(cls, path: str | Path) -> TargetSet:
        """Load from ``path`` (``.npz``) with ``allow_pickle=False``."""
        with np.load(path, allow_pickle=False) as f:
            split = str(f["split"].item()) if f["split"].ndim == 0 else str(f["split"])
            meta_str = (
                str(f["meta_json"].item())
                if f["meta_json"].ndim == 0
                else str(f["meta_json"])
            )
            return cls(
                split=split,
                seeds=np.asarray(f["seeds"], dtype=_DTYPE_SEED).copy(),
                target_pos=np.asarray(f["target_pos"], dtype=_DTYPE_TARGET).copy(),
                tip_to_target_init_distance=np.asarray(
                    f["tip_to_target_init_distance"], dtype=_DTYPE_DIST
                ).copy(),
                meta=json.loads(meta_str),
            )


# --- generation ---


def generate_seed_list(
    generator_seed: int, seed_offset: int, n: int
) -> np.ndarray:
    """Return ``[generator_seed + seed_offset + i for i in range(n)]`` as int64."""
    _check_non_negative_int(generator_seed, "generator_seed")
    _check_non_negative_int(seed_offset, "seed_offset")
    _check_positive_int(n, "n")
    base = generator_seed + seed_offset
    return np.arange(base, base + n, dtype=_DTYPE_SEED)


def _make_meta(
    config: TargetGenerationConfig,
    split: SplitConfig,
    site_name: str,
    site_low: tuple[float, float, float],
    site_high: tuple[float, float, float],
) -> dict[str, Any]:
    return {
        "env_id": config.env_id,
        "split": split.name,
        "generator_seed": config.generator_seed,
        "seed_offset": split.seed_offset,
        "n": split.n,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "generator_version": GENERATOR_VERSION,
        "source": "self_sampled_uniform_in_target_reach_range",
        "site_name": site_name,
        "site_low": list(map(float, site_low)),
        "site_high": list(map(float, site_high)),
        "workspace_bounds": None,  # reserved for future workspace override
        "notes": (
            "target_pos sampled by this module using "
            "np.random.default_rng(seed_i).uniform(low, high); env's "
            "internal np_random is not used because reset(seed=k) does "
            "not control target generation in MyoSuite ReachEnvV0."
        ),
    }


def generate_target_set(
    config: TargetGenerationConfig,
    split_name: str,
    env: Any | None = None,
) -> TargetSet:
    """Generate one ``TargetSet`` for ``split_name``.

    If ``env`` is None, one is constructed via ``make_env(config.env_id)``
    and closed at the end. Pass an existing env to amortize startup across
    multiple splits.
    """
    # Lazy MyoSuite import: keeps `import targets` cheap.
    import mujoco

    from myoarm_fse.envs.extractors import extract_state
    from myoarm_fse.envs.factory import make_env

    split = config.split(split_name)
    owns_env = env is None
    if owns_env:
        env = make_env(config.env_id)

    try:
        uw = env.unwrapped
        bbox = uw.target_reach_range
        if len(bbox) != 1:
            raise ValueError(
                f"expected exactly one target_reach_range entry, got {bbox!r}; "
                "multi-target envs are not yet supported"
            )
        site_name, (low_t, high_t) = next(iter(bbox.items()))
        low = np.asarray(low_t, dtype=np.float64)
        high = np.asarray(high_t, dtype=np.float64)
        target_sid = uw.mj_model.site(site_name + "_target").id

        seeds = generate_seed_list(
            config.generator_seed, split.seed_offset, split.n
        )
        target_pos = np.empty((split.n, _CART_DIM), dtype=_DTYPE_TARGET)
        init_dist = np.empty(split.n, dtype=_DTYPE_DIST)

        for i, seed in enumerate(seeds):
            rng = np.random.default_rng(int(seed))
            t = rng.uniform(low=low, high=high)  # float64 (3,)
            env.reset()
            uw.mj_model.site_pos[target_sid] = t
            mujoco.mj_forward(uw.mj_model, uw.mj_data)
            state = extract_state(env)
            target_pos[i] = state.target_pos
            init_dist[i] = float(np.linalg.norm(state.tip_pos - state.target_pos))

        meta = _make_meta(config, split, site_name, low_t, high_t)
        return TargetSet(
            split=split.name,
            seeds=seeds,
            target_pos=target_pos,
            tip_to_target_init_distance=init_dist,
            meta=meta,
        )
    finally:
        if owns_env:
            env.close()


def generate_all_target_sets(
    config: TargetGenerationConfig,
) -> dict[str, TargetSet]:
    """Generate every split in ``config`` reusing one env across splits."""
    from myoarm_fse.envs.factory import make_env

    env = make_env(config.env_id)
    try:
        return {
            s.name: generate_target_set(config, s.name, env=env)
            for s in config.splits
        }
    finally:
        env.close()
