"""``EpisodeLog`` — frozen schema for one episode of myoArm reaching.

Schema preserves every layer boundary established earlier (see
``02_InitialImplementationPlan.md``):

```text
neural_command  -> excitation_command -> excitation -> api_action -> last_ctrl
true MyoArmState (oracle)               obs MyoArmState (controller-facing)
```

All step-wise arrays are ragged: shape ``(T, ...)`` where ``T = n_steps``
is whatever the rollout actually walked. Padding is the dataset loader's
responsibility, not the logger's.

``meta`` is a JSON-serializable dict; on disk it lives in the npz as a
single ``meta_json`` string so files load with ``allow_pickle=False``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

_DT_F32 = np.dtype(np.float32)
_DT_INT = np.dtype(np.int64)
_DT_BOOL = np.dtype(np.bool_)
_CART_DIM: int = 3


# Names of the per-step arrays that share the ``(T, *)`` leading dimension.
# Tuple of (attr_name, expected_dtype, expected_inner_shape).
_STEP_FIELDS: tuple[tuple[str, np.dtype, tuple[int, ...] | None], ...] = (
    ("step", _DT_INT, ()),
    ("time", _DT_F32, ()),
    ("true_qpos", _DT_F32, None),       # inner dim = qpos_dim (env-specific)
    ("true_qvel", _DT_F32, None),
    ("true_act", _DT_F32, None),
    ("true_tip_pos", _DT_F32, (_CART_DIM,)),
    ("true_target_pos", _DT_F32, (_CART_DIM,)),
    ("true_reach_err", _DT_F32, (_CART_DIM,)),
    ("obs_qpos", _DT_F32, None),
    ("obs_qvel", _DT_F32, None),
    ("obs_act", _DT_F32, None),
    ("obs_tip_pos", _DT_F32, (_CART_DIM,)),
    ("obs_target_pos", _DT_F32, (_CART_DIM,)),
    ("obs_reach_err", _DT_F32, (_CART_DIM,)),
    ("neural_command", _DT_F32, None),  # inner dim = action_dim
    ("excitation_command", _DT_F32, None),
    ("excitation", _DT_F32, None),
    ("api_action", _DT_F32, None),
    ("last_ctrl", _DT_F32, None),
    ("reward", _DT_F32, ()),
    ("terminated", _DT_BOOL, ()),
    ("truncated", _DT_BOOL, ()),
)


@dataclass(frozen=True, eq=False)
class EpisodeLog:
    """One episode worth of recorded trajectories + metadata."""

    # Episode-level metadata (single values).
    episode_id: int
    target_id: str
    target_split: str
    target_seed: int
    target_pos_set: np.ndarray  # float32 (3,)
    controller_name: str
    controller_seed: int
    sdn_sigma: float
    sdn_seed: int
    obs_noise_sigma: dict[str, float]
    obs_noise_seed: int
    obs_delay_steps: int
    obs_compose: str
    max_steps: int
    n_steps: int
    created_at: str
    config_hash: str

    # Step-wise arrays, all length T = n_steps.
    step: np.ndarray
    time: np.ndarray
    true_qpos: np.ndarray
    true_qvel: np.ndarray
    true_act: np.ndarray
    true_tip_pos: np.ndarray
    true_target_pos: np.ndarray
    true_reach_err: np.ndarray
    obs_qpos: np.ndarray
    obs_qvel: np.ndarray
    obs_act: np.ndarray
    obs_tip_pos: np.ndarray
    obs_target_pos: np.ndarray
    obs_reach_err: np.ndarray
    neural_command: np.ndarray
    excitation_command: np.ndarray
    excitation: np.ndarray
    api_action: np.ndarray
    last_ctrl: np.ndarray
    reward: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray

    # Free-form extras (must be JSON-serializable).
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Scalar metadata sanity (light-touch; constructors of upstream
        # callers do their own deeper validation).
        if self.n_steps < 0:
            raise ValueError(f"n_steps must be >= 0, got {self.n_steps}")
        if self.max_steps < self.n_steps:
            raise ValueError(
                f"max_steps={self.max_steps} must be >= n_steps={self.n_steps}"
            )
        if self.target_pos_set.shape != (_CART_DIM,):
            raise ValueError(
                f"target_pos_set must have shape (3,), got {self.target_pos_set.shape}"
            )
        if self.target_pos_set.dtype != _DT_F32:
            raise ValueError(
                f"target_pos_set must be float32, got {self.target_pos_set.dtype}"
            )

        # Step arrays: all length T, dtype/shape per the schema.
        T = self.n_steps
        for name, expected_dtype, expected_inner in _STEP_FIELDS:
            arr = getattr(self, name)
            if not isinstance(arr, np.ndarray):
                raise ValueError(
                    f"{name} must be np.ndarray, got {type(arr).__name__}"
                )
            if arr.dtype != expected_dtype:
                raise ValueError(
                    f"{name} must be {expected_dtype}, got {arr.dtype}"
                )
            if arr.shape[0] != T:
                raise ValueError(
                    f"{name}.shape[0] must equal n_steps={T}, "
                    f"got {arr.shape[0]} (full shape {arr.shape})"
                )
            if expected_inner is not None:
                if arr.shape[1:] != expected_inner:
                    raise ValueError(
                        f"{name}.shape[1:] must be {expected_inner}, "
                        f"got {arr.shape[1:]}"
                    )

        # JSON-serializability of dict-like metadata, fail-fast.
        try:
            json.dumps(self.obs_noise_sigma)
            json.dumps(self.meta)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"meta or obs_noise_sigma not JSON-serializable: {exc}") from exc

    # --- save / load ---

    _SCALAR_META_FIELDS: ClassVar[tuple[str, ...]] = (
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
    )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta_payload = {
            **{k: getattr(self, k) for k in self._SCALAR_META_FIELDS},
            "obs_noise_sigma": self.obs_noise_sigma,
            "meta": self.meta,
        }
        meta_json = json.dumps(meta_payload)
        np.savez(
            path,
            meta_json=np.array(meta_json),
            target_pos_set=self.target_pos_set,
            **{name: getattr(self, name) for name, _, _ in _STEP_FIELDS},
        )

    @classmethod
    def load(cls, path: str | Path) -> EpisodeLog:
        with np.load(path, allow_pickle=False) as f:
            meta_str = (
                str(f["meta_json"].item())
                if f["meta_json"].ndim == 0
                else str(f["meta_json"])
            )
            meta_payload = json.loads(meta_str)
            kwargs: dict[str, Any] = {}
            for k in cls._SCALAR_META_FIELDS:
                kwargs[k] = meta_payload[k]
            kwargs["obs_noise_sigma"] = meta_payload["obs_noise_sigma"]
            kwargs["meta"] = meta_payload.get("meta", {})
            kwargs["target_pos_set"] = np.asarray(
                f["target_pos_set"], dtype=_DT_F32
            ).copy()
            for name, dtype, _ in _STEP_FIELDS:
                kwargs[name] = np.asarray(f[name], dtype=dtype).copy()
        return cls(**kwargs)

    # --- helpers ---

    @property
    def action_dim(self) -> int:
        if self.api_action.ndim < 2:
            return 0
        return int(self.api_action.shape[1])
