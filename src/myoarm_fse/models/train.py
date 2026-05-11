"""Training loop, save/load, rollout evaluation for ``ForwardMLP``.

Pipeline (intended use from ``scripts/train_forward_model.py``):

1. ``setup_seeds(master_seed)`` to derive child seeds and seed
   numpy / torch / random.
2. ``make_train_val_split(dataset, val_step=...)`` to hold out every
   N-th source episode for validation.
3. ``ForwardMLP(state_dim, action_dim, hidden_dims)``.
4. ``train_forward_model(model, train_ds, val_ds, config, *, seeds)``
   to fit. Returns the best-by-val-loss model and a metrics dict
   containing the loss curves and seed record.
5. ``rollout_predictions(model, eval_ds, horizons=...)`` for one-step
   and rollout MSE / tip prediction error per horizon.
6. ``save_model(model, config, metrics, *, path)`` to persist the
   model directory.
7. ``load_model(path)`` to round-trip.

Loss is MSE on Δx (residual parameterization). Optimizer Adam,
batch_size 256, no scheduler, early stopping on val loss with
patience 20. CPU determinism only — ``torch.use_deterministic_algorithms``
is intentionally not toggled (CUDA is not the Phase 0 target).
"""

from __future__ import annotations

import copy
import json
import random
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from myoarm_fse.envs.state import StateSpec
from myoarm_fse.metrics.prediction import (
    one_step_prediction_mse,
    rollout_mse,
    tip_prediction_error,
)
from myoarm_fse.models.datasets import (
    TransitionDataset,
    split_by_local_indices,
)
from myoarm_fse.models.mlp import ForwardMLP


# --- config ---


@dataclass(frozen=True)
class TrainConfig:
    """Hyperparameters for ``train_forward_model``."""

    optimizer: str = "adam"
    lr: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 256
    epochs: int = 200
    early_stopping_patience: int = 20
    early_stopping_min_delta: float = 0.0
    grad_clip: float | None = None
    seed: int = 0
    val_step: int = 5
    # Multi-step rollout supervision: 1 = single-step Δx loss (default,
    # Phase 0). ``multi_step >= 2`` switches to a rollout loss where the
    # model predicts K consecutive steps starting from an in-episode
    # anchor and MSE is summed over those K predictions, dividing by K
    # for scale parity with the single-step path.
    multi_step: int = 1

    def __post_init__(self) -> None:
        if self.optimizer not in ("adam", "adamw"):
            raise ValueError(
                f"optimizer must be 'adam' or 'adamw', got {self.optimizer!r}"
            )
        for name in ("batch_size", "epochs", "early_stopping_patience",
                     "val_step", "multi_step"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, int) or v <= 0:
                raise ValueError(f"{name} must be a positive int, got {v!r}")
        for name in ("lr", "weight_decay", "early_stopping_min_delta"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or v < 0.0 or not np.isfinite(v):
                raise ValueError(
                    f"{name} must be a non-negative finite number, got {v!r}"
                )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainConfig:
        valid = {f.name for f in fields(cls)}
        unknown = set(data.keys()) - valid
        if unknown:
            raise ValueError(f"unknown TrainConfig keys: {sorted(unknown)}")
        return cls(**data)


# --- seeds ---


def setup_seeds(master_seed: int) -> dict[str, int]:
    """Derive child seeds for model init / dataset shuffle / dataloader.

    Seeds the global numpy / torch / random RNGs from the model_init
    child seed and returns the dict so the caller can pass the rest
    explicitly to data shufflers / DataLoaders.
    """
    if isinstance(master_seed, bool) or not isinstance(master_seed, int):
        raise ValueError(
            f"master_seed must be int, got {type(master_seed).__name__}"
        )
    ss = np.random.SeedSequence(master_seed)
    children = ss.spawn(3)
    seeds = {
        "model_init": int(children[0].generate_state(1)[0]),
        "dataset_shuffle": int(children[1].generate_state(1)[0]),
        "dataloader": int(children[2].generate_state(1)[0]),
    }
    np.random.seed(seeds["model_init"])
    torch.manual_seed(seeds["model_init"])
    random.seed(seeds["model_init"])
    return seeds


# --- val split ---


def make_train_val_split(
    dataset: TransitionDataset, *, val_step: int = 5
) -> tuple[TransitionDataset, TransitionDataset]:
    """Hold out every ``val_step``-th source episode for validation.

    Splits by *local position* in ``episode_metadata`` so the function
    behaves correctly even when episode_id values collide (e.g. after
    concatenating multiple source runs that each number their episodes
    from 0). The one at local index ``0, val_step, 2*val_step, …``
    goes to val; the rest goes to train. Original ``episode_id`` is
    preserved in ``episode_metadata[i]["episode_id"]``; the split's
    local ``episode_index`` is reindexed from 0.
    """
    if isinstance(val_step, bool) or not isinstance(val_step, int) or val_step <= 0:
        raise ValueError(f"val_step must be a positive int, got {val_step!r}")
    val_indices = [i for i in range(dataset.n_episodes) if i % val_step == 0]
    return split_by_local_indices(dataset, val_indices=val_indices)


# --- minibatch ---


def _iter_minibatches(
    n: int, batch_size: int, *, shuffle: bool, rng: np.random.Generator | None
) -> Iterator[np.ndarray]:
    if n == 0:
        return
    if shuffle:
        if rng is None:
            rng = np.random.default_rng()
        idx = rng.permutation(n)
    else:
        idx = np.arange(n)
    for start in range(0, n, batch_size):
        yield idx[start : start + batch_size]


def _valid_multistep_starts(
    episode_index: np.ndarray, K: int
) -> np.ndarray:
    """Indices ``i`` such that ``i, i+1, ..., i+K-1`` all share the same episode_index.

    Used to build the set of valid K-step rollout anchors. ``episode_index``
    is monotone non-decreasing in :class:`TransitionDataset` (transitions
    are appended in episode order), so a chunk is valid iff
    ``episode_index[i] == episode_index[i + K - 1]``.
    """
    n = episode_index.shape[0]
    if K <= 0:
        raise ValueError(f"K must be > 0, got {K}")
    if n < K:
        return np.empty(0, dtype=np.int64)
    end = n - K + 1
    head = episode_index[:end]
    tail = episode_index[K - 1 : K - 1 + end]
    return np.where(head == tail)[0].astype(np.int64)


def _epoch_loss(
    model: ForwardMLP,
    ds: TransitionDataset,
    *,
    batch_size: int,
    optimizer: torch.optim.Optimizer | None,
    rng: np.random.Generator | None,
    grad_clip: float | None,
    multi_step: int = 1,
    valid_starts: np.ndarray | None = None,
) -> float:
    """Run one epoch of MSE loss.

    ``multi_step == 1`` uses the Phase 0 single-step Δx loss on every
    transition; ``multi_step >= 2`` uses a K-step rollout loss over
    pre-computed ``valid_starts`` indices, averaging MSE across the K
    prediction steps.
    """
    is_train = optimizer is not None
    model.train(is_train)
    if ds.n == 0:
        return 0.0

    if multi_step <= 1:
        x_t = torch.from_numpy(ds.x)
        u_t = torch.from_numpy(ds.u)
        dx_t = torch.from_numpy(ds.dx)
        total = 0.0
        counted = 0
        loss_fn = nn.MSELoss(reduction="sum")
        for idx in _iter_minibatches(ds.n, batch_size, shuffle=is_train, rng=rng):
            idx_t = torch.from_numpy(idx).long()
            xb = x_t[idx_t]
            ub = u_t[idx_t]
            dxb = dx_t[idx_t]
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                pred = model(xb, ub)
                loss = loss_fn(pred, dxb) / dxb.numel()
                loss.backward()
                if grad_clip is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            else:
                with torch.no_grad():
                    pred = model(xb, ub)
                    loss = loss_fn(pred, dxb) / dxb.numel()
            total += float(loss.item()) * dxb.numel()
            counted += dxb.numel()
        return total / counted if counted > 0 else 0.0

    # K-step rollout loss.
    K = int(multi_step)
    if valid_starts is None:
        valid_starts = _valid_multistep_starts(ds.episode_index, K)
    n_valid = int(valid_starts.shape[0])
    if n_valid == 0:
        return 0.0

    x_all = torch.from_numpy(ds.x)
    u_all = torch.from_numpy(ds.u)
    x_next_all = torch.from_numpy(ds.x_next)
    loss_fn = nn.MSELoss(reduction="sum")
    total = 0.0
    counted = 0
    for batch_idx in _iter_minibatches(
        n_valid, batch_size, shuffle=is_train, rng=rng,
    ):
        starts_np = valid_starts[batch_idx]
        starts = torch.from_numpy(starts_np).long()
        # Initial state for each chunk in the minibatch.
        x_curr = x_all[starts]                 # (B, state_dim)
        chunk_loss = torch.zeros((), dtype=x_curr.dtype)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        for k in range(K):
            u_k = u_all[starts + k]
            x_true_k = x_next_all[starts + k]  # = x_{i+k+1}
            if is_train:
                dx_pred = model(x_curr, u_k)
            else:
                with torch.no_grad():
                    dx_pred = model(x_curr, u_k)
            x_pred = x_curr + dx_pred
            # Per-step MSE; sum across K and normalize at the end.
            chunk_loss = chunk_loss + loss_fn(x_pred, x_true_k)
            # Free-running rollout: use the model's own prediction as the next prior.
            x_curr = x_pred if is_train else x_pred.detach()
        n_elems = K * starts.shape[0] * x_all.shape[1]
        loss = chunk_loss / n_elems
        if is_train:
            loss.backward()
            if grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        total += float(loss.item()) * n_elems
        counted += n_elems
    return total / counted if counted > 0 else 0.0


# --- training ---


def _build_optimizer(
    model: nn.Module, config: TrainConfig
) -> torch.optim.Optimizer:
    if config.optimizer == "adam":
        return torch.optim.Adam(
            model.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )
    if config.optimizer == "adamw":
        return torch.optim.AdamW(
            model.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )
    raise ValueError(f"unsupported optimizer: {config.optimizer!r}")


def train_forward_model(
    model: ForwardMLP,
    train_ds: TransitionDataset,
    val_ds: TransitionDataset,
    config: TrainConfig,
    *,
    seeds: dict[str, int] | None = None,
) -> tuple[ForwardMLP, dict[str, Any]]:
    """Fit ``model`` to ``(x, u, dx)`` triples in ``train_ds``.

    Returns ``(best_model, metrics)`` where ``best_model`` has the
    state_dict from the epoch with lowest val loss, and ``metrics``
    holds the loss curves, best epoch, seed record, and final
    train/val loss.
    """
    if not isinstance(model, ForwardMLP):
        raise ValueError(
            f"model must be ForwardMLP, got {type(model).__name__}"
        )
    if not isinstance(train_ds, TransitionDataset):
        raise ValueError("train_ds must be TransitionDataset")
    if not isinstance(val_ds, TransitionDataset):
        raise ValueError("val_ds must be TransitionDataset")
    if train_ds.state_dim != model.state_dim or train_ds.action_dim != model.action_dim:
        raise ValueError(
            f"train_ds dims ({train_ds.state_dim}, {train_ds.action_dim}) "
            f"do not match model ({model.state_dim}, {model.action_dim})"
        )

    optimizer = _build_optimizer(model, config)
    shuffle_rng = np.random.default_rng(
        seeds["dataset_shuffle"] if seeds and "dataset_shuffle" in seeds else None
    )

    # Pre-compute valid K-step anchor indices so each epoch only does
    # the cheap shuffle over them.
    K = int(config.multi_step)
    if K > 1:
        train_starts = _valid_multistep_starts(train_ds.episode_index, K)
        val_starts = _valid_multistep_starts(val_ds.episode_index, K)
    else:
        train_starts = None
        val_starts = None

    train_history: list[float] = []
    val_history: list[float] = []
    best_val = float("inf")
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    epochs_since_improve = 0

    for epoch in range(config.epochs):
        train_loss = _epoch_loss(
            model, train_ds,
            batch_size=config.batch_size,
            optimizer=optimizer,
            rng=shuffle_rng,
            grad_clip=config.grad_clip,
            multi_step=K,
            valid_starts=train_starts,
        )
        val_loss = _epoch_loss(
            model, val_ds,
            batch_size=config.batch_size,
            optimizer=None, rng=None, grad_clip=None,
            multi_step=K,
            valid_starts=val_starts,
        )
        train_history.append(train_loss)
        val_history.append(val_loss)

        improved = val_loss < best_val - config.early_stopping_min_delta
        if improved:
            best_val = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1
            if epochs_since_improve >= config.early_stopping_patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    metrics: dict[str, Any] = {
        "train_loss_history": train_history,
        "val_loss_history": val_history,
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val) if best_epoch >= 0 else float("inf"),
        "stopped_early": (
            best_epoch >= 0 and len(train_history) < config.epochs
        ),
        "seeds": dict(seeds) if seeds is not None else {},
        "n_train": int(train_ds.n),
        "n_val": int(val_ds.n),
    }
    return model, metrics


# --- rollout evaluation ---


def _state_layout(state_dim: int) -> dict[str, slice]:
    """Recover ``MyoArmState`` layout from a flat dim of 83 (myoArm reach).

    Falls back to "tip_pos last 9 fields are the cartesian triples" if
    the dim does not match — this keeps the helper robust to small
    schema changes while still extracting tip_pos for the prediction
    metric.
    """
    # Defer to StateSpec by inferring qpos/qvel/act dims from a known total.
    # For myoArm reach state_dim = qpos + qvel + act + 9 with qpos = qvel.
    # We infer qpos = qvel and act from the 83-total: qpos*2 + act + 9 = 83.
    # In practice the Phase 0 build_transitions produces state_dim=83 only.
    cart_total = 9  # tip_pos + target_pos + reach_err
    if state_dim < cart_total + 2:
        raise ValueError(
            f"state_dim={state_dim} is too small to contain qpos/qvel/act + 9 cart"
        )
    # Heuristic for myoArm reach: qpos == qvel == 20, act == 34.
    if state_dim == 83:
        spec = StateSpec(qpos_dim=20, qvel_dim=20, act_dim=34)
        return spec.layout()
    # Fallback: only tip_pos slice is needed for the prediction metric;
    # construct a minimal layout from the cartesian tail.
    tip_start = state_dim - cart_total
    return {
        "tip_pos": slice(tip_start, tip_start + 3),
        "target_pos": slice(tip_start + 3, tip_start + 6),
        "reach_err": slice(tip_start + 6, tip_start + 9),
    }


def _episode_slices(
    episode_index: np.ndarray, n_episodes: int
) -> list[slice]:
    """Return the [start, end) slice for each contiguous episode block.

    Assumes ``episode_index`` is non-decreasing (true after
    ``build_transitions`` and after ``split_by_episode``; not after
    ``shuffle_transitions``).
    """
    slices: list[slice] = []
    for ep in range(n_episodes):
        mask = episode_index == ep
        if not mask.any():
            slices.append(slice(0, 0))
            continue
        idx = np.nonzero(mask)[0]
        slices.append(slice(int(idx[0]), int(idx[-1] + 1)))
    return slices


def rollout_predictions(
    model: ForwardMLP,
    dataset: TransitionDataset,
    *,
    horizons: tuple[int, ...] = (1, 10, 50),
) -> dict[int, dict[str, float]]:
    """Closed-loop autoregressive rollout MSE per horizon.

    For each horizon ``h``, slide a length-``h`` window over each
    contiguous episode block. At each window start ``t``, seed the
    rollout with ``x[t]`` and step the model forward ``h`` times using
    the recorded actions ``u[t..t+h-1]``. Compare the predicted state
    trajectory to the ground-truth states ``x[t+1..t+h]`` (which the
    dataset stores as ``x_next[t..t+h-1]``).

    Returns ``{h: {"rollout_mse": ..., "tip_prediction_error": ...}}``.
    Empty-input or impossible (h > min episode length) horizons report
    ``0.0`` / ``inf`` respectively (see code).
    """
    if not isinstance(model, ForwardMLP):
        raise ValueError("model must be ForwardMLP")
    if not isinstance(dataset, TransitionDataset):
        raise ValueError("dataset must be TransitionDataset")
    if dataset.n == 0:
        return {h: {"rollout_mse": 0.0, "tip_prediction_error": 0.0} for h in horizons}

    layout = _state_layout(dataset.state_dim)
    tip_slice = layout["tip_pos"]

    ep_slices = _episode_slices(dataset.episode_index, dataset.n_episodes)
    model.eval()

    results: dict[int, dict[str, float]] = {}
    for h in horizons:
        if isinstance(h, bool) or not isinstance(h, int) or h <= 0:
            raise ValueError(f"each horizon must be a positive int, got {h!r}")
        true_traj_chunks: list[np.ndarray] = []
        pred_traj_chunks: list[np.ndarray] = []
        for ep_slice in ep_slices:
            ep_n = ep_slice.stop - ep_slice.start
            if ep_n < h:
                continue
            x_ep = dataset.x[ep_slice]            # (ep_n, state_dim)
            u_ep = dataset.u[ep_slice]            # (ep_n, action_dim)
            x_next_ep = dataset.x_next[ep_slice]  # (ep_n, state_dim)
            n_starts = ep_n - h + 1
            # For each window start t, predict x_{t+1}, x_{t+2}, ..., x_{t+h}.
            # Ground truth is x_next_ep[t : t+h] (shape (h, state_dim)).
            true_chunk = np.empty((n_starts * h, dataset.state_dim), dtype=np.float32)
            pred_chunk = np.empty((n_starts * h, dataset.state_dim), dtype=np.float32)
            with torch.no_grad():
                for k, t in enumerate(range(n_starts)):
                    # Seed rollout
                    x = torch.from_numpy(x_ep[t]).unsqueeze(0)  # (1, state_dim)
                    pred_steps = []
                    for j in range(h):
                        u = torch.from_numpy(u_ep[t + j]).unsqueeze(0)
                        x = x + model(x, u)
                        pred_steps.append(x.squeeze(0).numpy())
                    true_chunk[k * h : (k + 1) * h] = x_next_ep[t : t + h]
                    pred_chunk[k * h : (k + 1) * h] = np.stack(pred_steps, axis=0)
            true_traj_chunks.append(true_chunk)
            pred_traj_chunks.append(pred_chunk)
        if not true_traj_chunks:
            results[h] = {"rollout_mse": 0.0, "tip_prediction_error": 0.0}
            continue
        true_traj = np.concatenate(true_traj_chunks, axis=0)
        pred_traj = np.concatenate(pred_traj_chunks, axis=0)
        # rollout_mse / tip_prediction_error from Step 7.
        if h == 1:
            mse = one_step_prediction_mse(true_traj, pred_traj)
        else:
            mse = rollout_mse(true_traj, pred_traj)
        tip_err = tip_prediction_error(
            true_traj[:, tip_slice], pred_traj[:, tip_slice]
        )
        results[h] = {
            "rollout_mse": float(mse),
            "tip_prediction_error": float(tip_err),
        }
    return results


# --- save / load ---


def save_model(
    model: ForwardMLP,
    config: dict[str, Any],
    metrics: dict[str, Any],
    *,
    path: str | Path,
    info: dict[str, Any] | None = None,
) -> None:
    """Persist ``state_dict`` + ``config.json`` + ``metrics.json`` (+ ``info.json``)."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), p / "model.pt")
    (p / "config.json").write_text(json.dumps(config, indent=2))
    (p / "metrics.json").write_text(json.dumps(metrics, indent=2))
    info_payload = {
        "model_id": p.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if info:
        info_payload.update(info)
    (p / "info.json").write_text(json.dumps(info_payload, indent=2))


def load_model(
    path: str | Path,
) -> tuple[ForwardMLP, dict[str, Any], dict[str, Any]]:
    """Inverse of :func:`save_model`. Returns ``(model, config, metrics)``."""
    p = Path(path)
    config = json.loads((p / "config.json").read_text())
    arch = config["architecture"]
    model = ForwardMLP(
        state_dim=int(arch["state_dim"]),
        action_dim=int(arch["action_dim"]),
        hidden_dims=tuple(int(h) for h in arch["hidden_dims"]),
    )
    state_dict = torch.load(p / "model.pt", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    metrics = json.loads((p / "metrics.json").read_text())
    return model, config, metrics


def make_model_id(now: datetime | None = None) -> str:
    """Filesystem-safe UTC timestamp, matching the Step 5 run_id format."""
    t = now if now is not None else datetime.now(timezone.utc)
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    return t.astimezone(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
