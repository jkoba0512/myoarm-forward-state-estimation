"""Transition dataset for forward-model training.

Builds ``(x_t, u_t, x_{t+1}, Δx_t)`` tuples from a sequence of
``EpisodeLog`` objects. Layer choices follow the Step 8 design notes:

- ``x`` is the full ``MyoArmState.flatten()`` vector
  (``qpos | qvel | act | tip_pos | target_pos | reach_err``;
  ``state_dim = 83`` for myoArm reach).
- ``u`` is the post-SDN ``excitation`` (Step 3 canonical research-side
  input; not ``excitation_command`` and not ``last_ctrl``).
- The last step of every episode is dropped (no ``x_{t+1}``).
- Episodes with ``n_steps < 2`` contribute zero transitions.
- Mid-episode ``terminated`` / ``truncated`` flags are an invariant
  violation (rollout breaks on termination); ``build_transitions``
  raises ``ValueError`` if it sees one.

The dataset is flat-concatenated across episodes
(``N = Σ_i (n_steps_i - 1)``) with an ``episode_index`` column
preserving origin so train/val splits and per-episode evaluation
remain possible. ``episode_metadata`` carries per-episode provenance
(target_id, controller_name, sdn_sigma, …) keyed by the local
``episode_index`` rather than per-row.

Save/load uses ``np.savez`` with ``allow_pickle=False``;
``episode_metadata`` is JSON-encoded into a single ``meta_json`` array.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from myoarm_fse.data.schema import EpisodeLog
from myoarm_fse.envs.state import MyoArmState

_DT_F32 = np.dtype(np.float32)
_DT_INT = np.dtype(np.int64)
_DX_ATOL: float = 1e-5

# Field order in MyoArmState.flatten(); kept here for clarity.
_STATE_FIELDS: tuple[str, ...] = (
    "qpos",
    "qvel",
    "act",
    "tip_pos",
    "target_pos",
    "reach_err",
)


def _check_non_negative_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be int, got bool: {value!r}")
    if not isinstance(value, int):
        raise ValueError(
            f"{name} must be int, got {type(value).__name__}: {value!r}"
        )
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")
    return value


# --- Dataset ---


@dataclass(frozen=True, eq=False)
class TransitionDataset:
    """Flat-concatenated forward-dynamics transitions from one or more episodes.

    See module docstring for the layer choices behind each field.
    """

    x: np.ndarray
    u: np.ndarray
    x_next: np.ndarray
    dx: np.ndarray
    episode_index: np.ndarray
    state_dim: int
    action_dim: int
    n_episodes: int
    episode_metadata: tuple[dict[str, Any], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        _check_non_negative_int(self.state_dim, "state_dim")
        _check_non_negative_int(self.action_dim, "action_dim")
        _check_non_negative_int(self.n_episodes, "n_episodes")

        if not isinstance(self.episode_metadata, tuple):
            raise ValueError(
                f"episode_metadata must be tuple, "
                f"got {type(self.episode_metadata).__name__}"
            )
        if len(self.episode_metadata) != self.n_episodes:
            raise ValueError(
                f"episode_metadata length {len(self.episode_metadata)} "
                f"does not match n_episodes={self.n_episodes}"
            )
        for i, meta in enumerate(self.episode_metadata):
            if not isinstance(meta, dict):
                raise ValueError(
                    f"episode_metadata[{i}] must be dict, "
                    f"got {type(meta).__name__}"
                )

        # Array dtype / shape / finiteness.
        for name, expected_inner in (
            ("x", self.state_dim),
            ("u", self.action_dim),
            ("x_next", self.state_dim),
            ("dx", self.state_dim),
        ):
            arr = getattr(self, name)
            if not isinstance(arr, np.ndarray):
                raise ValueError(
                    f"{name} must be np.ndarray, got {type(arr).__name__}"
                )
            if arr.dtype != _DT_F32:
                raise ValueError(f"{name} must be float32, got {arr.dtype}")
            if arr.ndim != 2:
                raise ValueError(
                    f"{name} must be 2-D, got ndim={arr.ndim} shape={arr.shape}"
                )
            if arr.shape[1] != expected_inner:
                raise ValueError(
                    f"{name}.shape[1] must be {expected_inner}, got {arr.shape[1]}"
                )
            if not np.isfinite(arr).all():
                raise ValueError(f"{name} contains non-finite values")

        N = self.x.shape[0]
        if not (
            self.u.shape[0] == self.x_next.shape[0] == self.dx.shape[0] == N
        ):
            raise ValueError(
                f"x/u/x_next/dx must share the leading dim; got "
                f"{self.x.shape[0]}/{self.u.shape[0]}/"
                f"{self.x_next.shape[0]}/{self.dx.shape[0]}"
            )

        # episode_index shape / dtype / range.
        if not isinstance(self.episode_index, np.ndarray):
            raise ValueError(
                f"episode_index must be np.ndarray, "
                f"got {type(self.episode_index).__name__}"
            )
        if self.episode_index.dtype != _DT_INT:
            raise ValueError(
                f"episode_index must be int64, got {self.episode_index.dtype}"
            )
        if self.episode_index.shape != (N,):
            raise ValueError(
                f"episode_index must have shape ({N},), "
                f"got {self.episode_index.shape}"
            )
        if N > 0 and self.n_episodes > 0:
            if (self.episode_index < 0).any() or (
                self.episode_index >= self.n_episodes
            ).any():
                raise ValueError(
                    f"episode_index values must be in [0, {self.n_episodes}), "
                    f"got range [{int(self.episode_index.min())}, "
                    f"{int(self.episode_index.max())}]"
                )

        # dx consistency (defensive — build_transitions guarantees this,
        # but a caller constructing TransitionDataset directly might not).
        if N > 0 and not np.allclose(self.dx, self.x_next - self.x, atol=_DX_ATOL):
            raise ValueError(
                "dx must equal x_next - x (within float32 tolerance); "
                "use build_transitions to construct the dataset"
            )

        # episode_metadata JSON-serializability (fail fast at construction).
        try:
            json.dumps(list(self.episode_metadata))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"episode_metadata must be JSON-serializable: {exc}"
            ) from exc

    @property
    def n(self) -> int:
        return int(self.x.shape[0])

    # --- save / load ---

    def save(self, path: str | Path) -> None:
        """Save to ``path`` as ``.npz`` with JSON-encoded metadata."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta_json = json.dumps(list(self.episode_metadata))
        np.savez(
            path,
            x=self.x,
            u=self.u,
            x_next=self.x_next,
            dx=self.dx,
            episode_index=self.episode_index,
            state_dim=np.array(self.state_dim),
            action_dim=np.array(self.action_dim),
            n_episodes=np.array(self.n_episodes),
            meta_json=np.array(meta_json),
        )

    @classmethod
    def load(cls, path: str | Path) -> TransitionDataset:
        """Load from ``path`` (``.npz``) with ``allow_pickle=False``."""
        with np.load(path, allow_pickle=False) as f:
            meta_str = (
                str(f["meta_json"].item())
                if f["meta_json"].ndim == 0
                else str(f["meta_json"])
            )
            episode_metadata = tuple(json.loads(meta_str))
            return cls(
                x=np.asarray(f["x"], dtype=_DT_F32).copy(),
                u=np.asarray(f["u"], dtype=_DT_F32).copy(),
                x_next=np.asarray(f["x_next"], dtype=_DT_F32).copy(),
                dx=np.asarray(f["dx"], dtype=_DT_F32).copy(),
                episode_index=np.asarray(f["episode_index"], dtype=_DT_INT).copy(),
                state_dim=int(f["state_dim"]),
                action_dim=int(f["action_dim"]),
                n_episodes=int(f["n_episodes"]),
                episode_metadata=episode_metadata,
            )


# --- build / helpers ---


def _flatten_states(log: EpisodeLog) -> np.ndarray:
    """Return ``(T, state_dim)`` flat states from log's true_* fields.

    Concatenates ``true_qpos | true_qvel | true_act | true_tip_pos |
    true_target_pos | true_reach_err`` per step, matching the field order
    of ``MyoArmState.flatten()``.
    """
    T = log.n_steps
    if T == 0:
        # Determine state_dim from log's per-field shapes anyway, so the
        # caller can interrogate it without a loop body.
        state_dim = (
            log.true_qpos.shape[1]
            + log.true_qvel.shape[1]
            + log.true_act.shape[1]
            + 3 * 3  # tip_pos + target_pos + reach_err each (3,)
        )
        return np.empty((0, state_dim), dtype=_DT_F32)
    return np.concatenate(
        [
            log.true_qpos,
            log.true_qvel,
            log.true_act,
            log.true_tip_pos,
            log.true_target_pos,
            log.true_reach_err,
        ],
        axis=1,
        dtype=_DT_F32,
    )


def _episode_metadata_from_log(log: EpisodeLog) -> dict[str, Any]:
    return {
        "episode_id": int(log.episode_id),
        "target_id": str(log.target_id),
        "target_split": str(log.target_split),
        "target_seed": int(log.target_seed),
        "controller_name": str(log.controller_name),
        "controller_seed": int(log.controller_seed),
        "sdn_sigma": float(log.sdn_sigma),
        "sdn_seed": int(log.sdn_seed),
        "obs_noise_sigma": dict(log.obs_noise_sigma),
        "obs_noise_seed": int(log.obs_noise_seed),
        "obs_delay_steps": int(log.obs_delay_steps),
        "obs_compose": str(log.obs_compose),
        "n_steps": int(log.n_steps),
        "transitions_used": int(max(log.n_steps - 1, 0)),
        "config_hash": str(log.config_hash),
    }


def build_transitions(logs: Iterable[EpisodeLog]) -> TransitionDataset:
    """Build a ``TransitionDataset`` from ``EpisodeLog`` instances.

    Each episode contributes its ``T_i - 1`` non-terminal transitions:
    ``x_t = MyoArmState_t.flatten()``, ``u_t = excitation_t``,
    ``x_{t+1} = MyoArmState_{t+1}.flatten()``,
    ``Δx_t = x_{t+1} - x_t``. Episodes with ``n_steps < 2`` are skipped.
    Mid-episode ``terminated`` / ``truncated`` flags raise ``ValueError``.
    """
    xs: list[np.ndarray] = []
    us: list[np.ndarray] = []
    xnexts: list[np.ndarray] = []
    dxs: list[np.ndarray] = []
    eps: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []
    state_dim: int | None = None
    action_dim: int | None = None
    local_idx = 0

    for log in logs:
        if not isinstance(log, EpisodeLog):
            raise ValueError(
                f"all logs must be EpisodeLog, got {type(log).__name__}"
            )
        if log.n_steps < 2:
            continue
        # Defensive: rollout must break on termination, so flags before
        # the last step indicate corrupt data.
        if log.n_steps >= 2 and (
            log.terminated[:-1].any() or log.truncated[:-1].any()
        ):
            raise ValueError(
                f"episode {log.episode_id} has terminated/truncated set "
                "before the last step; rollout should break on termination"
            )

        x_full = _flatten_states(log)              # (T, state_dim)
        u_full = log.excitation                    # (T, action_dim)
        x_t = x_full[:-1].astype(_DT_F32, copy=True)
        u_t = u_full[:-1].astype(_DT_F32, copy=True)
        x_next = x_full[1:].astype(_DT_F32, copy=True)
        dx = (x_next - x_t).astype(_DT_F32, copy=False)

        if state_dim is None:
            state_dim = int(x_t.shape[1])
            action_dim = int(u_t.shape[1])
        else:
            if x_t.shape[1] != state_dim:
                raise ValueError(
                    f"state_dim mismatch across episodes: episode "
                    f"{log.episode_id} has {x_t.shape[1]}, expected {state_dim}"
                )
            if u_t.shape[1] != action_dim:
                raise ValueError(
                    f"action_dim mismatch across episodes: episode "
                    f"{log.episode_id} has {u_t.shape[1]}, expected {action_dim}"
                )

        xs.append(x_t)
        us.append(u_t)
        xnexts.append(x_next)
        dxs.append(dx)
        eps.append(np.full(x_t.shape[0], local_idx, dtype=_DT_INT))
        metadata.append(_episode_metadata_from_log(log))
        local_idx += 1

    if not xs:
        # No contributing episodes; return an empty but well-formed dataset.
        # If state_dim/action_dim are unknown, fall back to 0.
        return TransitionDataset(
            x=np.empty((0, state_dim or 0), dtype=_DT_F32),
            u=np.empty((0, action_dim or 0), dtype=_DT_F32),
            x_next=np.empty((0, state_dim or 0), dtype=_DT_F32),
            dx=np.empty((0, state_dim or 0), dtype=_DT_F32),
            episode_index=np.empty((0,), dtype=_DT_INT),
            state_dim=state_dim or 0,
            action_dim=action_dim or 0,
            n_episodes=0,
            episode_metadata=(),
        )

    return TransitionDataset(
        x=np.concatenate(xs, axis=0),
        u=np.concatenate(us, axis=0),
        x_next=np.concatenate(xnexts, axis=0),
        dx=np.concatenate(dxs, axis=0),
        episode_index=np.concatenate(eps, axis=0),
        state_dim=int(state_dim),
        action_dim=int(action_dim),
        n_episodes=len(metadata),
        episode_metadata=tuple(metadata),
    )


def shuffle_transitions(
    dataset: TransitionDataset,
    *,
    rng: np.random.Generator | int | None = None,
) -> TransitionDataset:
    """Return a new dataset with rows permuted.

    ``episode_index`` is permuted alongside ``x/u/x_next/dx`` so origin
    tracking is preserved. ``episode_metadata`` is unchanged (it is
    indexed by ``episode_index`` value, not by row).
    """
    if isinstance(rng, np.random.Generator):
        gen = rng
    elif rng is None:
        gen = np.random.default_rng()
    elif isinstance(rng, bool):
        raise TypeError(f"rng must be None / int / Generator, got bool: {rng!r}")
    elif isinstance(rng, int):
        gen = np.random.default_rng(rng)
    else:
        raise TypeError(
            f"rng must be None / int / Generator, got {type(rng).__name__}"
        )
    perm = gen.permutation(dataset.n)
    return TransitionDataset(
        x=dataset.x[perm].copy(),
        u=dataset.u[perm].copy(),
        x_next=dataset.x_next[perm].copy(),
        dx=dataset.dx[perm].copy(),
        episode_index=dataset.episode_index[perm].copy(),
        state_dim=dataset.state_dim,
        action_dim=dataset.action_dim,
        n_episodes=dataset.n_episodes,
        episode_metadata=dataset.episode_metadata,
    )


def split_by_local_indices(
    dataset: TransitionDataset,
    *,
    val_indices: Iterable[int],
) -> tuple[TransitionDataset, TransitionDataset]:
    """Partition ``dataset`` into ``(train, val)`` keyed by *local position*.

    ``val_indices`` are positions into ``dataset.episode_metadata`` (i.e.
    values in ``[0, dataset.n_episodes)``), NOT original episode_id
    values. Use this when episode_id is not guaranteed to be unique
    across the dataset (e.g. after concatenating multiple source runs).
    """
    indices = sorted({int(i) for i in val_indices})
    if any(i < 0 or i >= dataset.n_episodes for i in indices):
        raise ValueError(
            f"val_indices must lie in [0, {dataset.n_episodes}); got {indices}"
        )
    val_local = indices
    train_local = [i for i in range(dataset.n_episodes) if i not in set(val_local)]
    return (
        _select_episodes(dataset, train_local),
        _select_episodes(dataset, val_local),
    )


def split_by_episode(
    dataset: TransitionDataset,
    *,
    val_episode_ids: Iterable[int],
) -> tuple[TransitionDataset, TransitionDataset]:
    """Partition ``dataset`` into ``(train, val)`` keyed by ``episode_id``.

    ``val_episode_ids`` is the set of *original* ``episode_id`` values
    (as stored in ``episode_metadata[i]["episode_id"]``) to place in
    the val split. Episodes whose id is not in the set go to train.
    Both returned datasets have their ``episode_index`` reindexed
    starting from 0; ``episode_metadata`` is filtered and ordered to
    match.

    Raises ``ValueError`` if ``val_episode_ids`` contains an id not
    present in ``dataset.episode_metadata``.
    """
    val_ids = {int(i) for i in val_episode_ids}
    available_ids = {int(m["episode_id"]) for m in dataset.episode_metadata}
    unknown = val_ids - available_ids
    if unknown:
        raise ValueError(
            f"val_episode_ids contains ids not in dataset: {sorted(unknown)}"
        )

    train_local: list[int] = []
    val_local: list[int] = []
    for local_idx, meta in enumerate(dataset.episode_metadata):
        if int(meta["episode_id"]) in val_ids:
            val_local.append(local_idx)
        else:
            train_local.append(local_idx)

    return (
        _select_episodes(dataset, train_local),
        _select_episodes(dataset, val_local),
    )


def _select_episodes(
    dataset: TransitionDataset, keep_local: list[int]
) -> TransitionDataset:
    """Build a new dataset containing only the listed local episode indices."""
    if not keep_local:
        return TransitionDataset(
            x=np.empty((0, dataset.state_dim), dtype=_DT_F32),
            u=np.empty((0, dataset.action_dim), dtype=_DT_F32),
            x_next=np.empty((0, dataset.state_dim), dtype=_DT_F32),
            dx=np.empty((0, dataset.state_dim), dtype=_DT_F32),
            episode_index=np.empty((0,), dtype=_DT_INT),
            state_dim=dataset.state_dim,
            action_dim=dataset.action_dim,
            n_episodes=0,
            episode_metadata=(),
        )

    # Build mask + reindex map.
    keep_set = set(keep_local)
    row_mask = np.array(
        [int(idx) in keep_set for idx in dataset.episode_index],
        dtype=bool,
    )
    # local_idx → new_idx
    new_idx_map = {old: new for new, old in enumerate(keep_local)}
    new_episode_index = np.array(
        [new_idx_map[int(dataset.episode_index[i])] for i in np.nonzero(row_mask)[0]],
        dtype=_DT_INT,
    )
    return TransitionDataset(
        x=dataset.x[row_mask].copy(),
        u=dataset.u[row_mask].copy(),
        x_next=dataset.x_next[row_mask].copy(),
        dx=dataset.dx[row_mask].copy(),
        episode_index=new_episode_index,
        state_dim=dataset.state_dim,
        action_dim=dataset.action_dim,
        n_episodes=len(keep_local),
        episode_metadata=tuple(dataset.episode_metadata[i] for i in keep_local),
    )
