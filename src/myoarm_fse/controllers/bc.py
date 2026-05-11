"""Behavioral-cloning policy + controller wrapper.

Trains and loads a small MLP that maps a flat ``MyoArmState`` vector
(``state_dim``-dim) to a 34-dim muscle excitation in ``[0, 1]``. The
``BCController`` wraps a loaded ``BCPolicy`` so it satisfies the
``Controller`` protocol and can be dropped into
``run_closed_loop_episode`` in place of ScriptedReach / JointSpacePD /
HeuristicReach.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn

from myoarm_fse.envs.state import MyoArmState

_DTYPE: np.dtype = np.dtype(np.float32)
_LO: float = 0.0
_HI: float = 1.0


# --- model ---


class BCPolicy(nn.Module):
    """Small MLP regressor: flat state -> [0, 1]^action_dim."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int] = (256, 256),
    ) -> None:
        super().__init__()
        for name, val in (("state_dim", state_dim), ("action_dim", action_dim)):
            if isinstance(val, bool) or not isinstance(val, int) or val <= 0:
                raise ValueError(f"{name} must be a positive int, got {val!r}")
        for h in hidden_dims:
            if isinstance(h, bool) or not isinstance(h, int) or h <= 0:
                raise ValueError(
                    f"each hidden_dim must be a positive int, got {h!r}"
                )
        layers: list[nn.Module] = []
        prev = state_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, action_dim))
        self.net = nn.Sequential(*layers)
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.hidden_dims = tuple(int(h) for h in hidden_dims)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.state_dim:
            raise ValueError(
                f"x.shape[-1] must be {self.state_dim}, got {x.shape[-1]}"
            )
        logits = self.net(x)
        return torch.sigmoid(logits)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# --- training ---


@dataclass(frozen=True)
class BCTrainConfig:
    optimizer: str = "adam"
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 256
    epochs: int = 50
    seed: int = 0
    val_fraction: float = 0.1

    def __post_init__(self) -> None:
        if self.optimizer not in ("adam", "adamw"):
            raise ValueError(
                f"optimizer must be 'adam' or 'adamw', got {self.optimizer!r}"
            )
        for name in ("batch_size", "epochs"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, int) or v <= 0:
                raise ValueError(f"{name} must be a positive int, got {v!r}")
        for name in ("lr", "weight_decay", "val_fraction"):
            v = getattr(self, name)
            if (
                isinstance(v, bool) or not isinstance(v, (int, float))
                or v < 0.0 or not np.isfinite(v)
            ):
                raise ValueError(
                    f"{name} must be a non-negative finite number, got {v!r}"
                )
        if self.val_fraction >= 1.0:
            raise ValueError(
                f"val_fraction must be < 1.0, got {self.val_fraction}"
            )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BCTrainConfig":
        valid = {f.name for f in fields(cls)}
        unknown = set(data.keys()) - valid
        if unknown:
            raise ValueError(f"unknown BCTrainConfig keys: {sorted(unknown)}")
        return cls(**data)


def train_bc_policy(
    states: np.ndarray,
    actions: np.ndarray,
    *,
    config: BCTrainConfig,
    hidden_dims: Sequence[int] = (256, 256),
) -> tuple[BCPolicy, dict[str, Any]]:
    if states.ndim != 2 or actions.ndim != 2:
        raise ValueError("states and actions must be 2-D")
    if states.shape[0] != actions.shape[0]:
        raise ValueError("states and actions must have matching row count")
    n = states.shape[0]
    if n == 0:
        raise ValueError("states/actions must be non-empty")

    state_dim = int(states.shape[1])
    action_dim = int(actions.shape[1])

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    rng = np.random.default_rng(config.seed)
    perm = rng.permutation(n)
    n_val = int(round(n * config.val_fraction))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    x_tr = torch.from_numpy(states[train_idx].astype(np.float32))
    y_tr = torch.from_numpy(actions[train_idx].astype(np.float32))
    x_val = torch.from_numpy(states[val_idx].astype(np.float32))
    y_val = torch.from_numpy(actions[val_idx].astype(np.float32))

    model = BCPolicy(state_dim=state_dim, action_dim=action_dim, hidden_dims=hidden_dims)
    if config.optimizer == "adam":
        opt = torch.optim.Adam(
            model.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )
    else:
        opt = torch.optim.AdamW(
            model.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )
    loss_fn = nn.MSELoss()
    history: list[dict[str, float]] = []
    train_n = x_tr.shape[0]
    model.train()
    for ep in range(config.epochs):
        perm_ep = rng.permutation(train_n)
        running = 0.0
        running_n = 0
        for start in range(0, train_n, config.batch_size):
            idx = perm_ep[start : start + config.batch_size]
            xb = x_tr[idx]
            yb = y_tr[idx]
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            running += float(loss.item()) * xb.shape[0]
            running_n += xb.shape[0]
        model.eval()
        with torch.no_grad():
            val_pred = model(x_val) if n_val > 0 else None
            val_loss = float(loss_fn(val_pred, y_val).item()) if val_pred is not None else 0.0
        model.train()
        history.append({
            "epoch": ep,
            "train_loss": running / max(running_n, 1),
            "val_loss": val_loss,
        })
    model.eval()
    return model, {
        "n_samples": int(n),
        "n_train": int(train_idx.shape[0]),
        "n_val": int(n_val),
        "state_dim": int(state_dim),
        "action_dim": int(action_dim),
        "history": history,
        "final_train_loss": history[-1]["train_loss"],
        "final_val_loss": history[-1]["val_loss"],
    }


# --- save / load ---


def make_bc_model_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


def save_bc_policy(
    model: BCPolicy,
    config: dict[str, Any],
    metrics: dict[str, Any],
    *,
    path: str | Path,
) -> None:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), p / "model.pt")
    (p / "config.json").write_text(json.dumps(config, indent=2))
    (p / "metrics.json").write_text(json.dumps(metrics, indent=2))


def load_bc_policy(
    path: str | Path,
) -> tuple[BCPolicy, dict[str, Any], dict[str, Any]]:
    p = Path(path)
    config = json.loads((p / "config.json").read_text())
    arch = config.get("architecture", {})
    state_dim = int(arch["state_dim"])
    action_dim = int(arch["action_dim"])
    hidden_dims = tuple(int(h) for h in arch.get("hidden_dims", (256, 256)))
    model = BCPolicy(state_dim=state_dim, action_dim=action_dim, hidden_dims=hidden_dims)
    state_dict = torch.load(p / "model.pt", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    metrics = json.loads((p / "metrics.json").read_text())
    return model, config, metrics


# --- controller wrapper ---


class BCController:
    """Wraps a trained ``BCPolicy`` into a closed-loop ``Controller``."""

    def __init__(self, policy: BCPolicy, action_dim: int) -> None:
        if not isinstance(policy, BCPolicy):
            raise TypeError(
                f"policy must be BCPolicy, got {type(policy).__name__}"
            )
        if policy.action_dim != action_dim:
            raise ValueError(
                f"policy.action_dim={policy.action_dim} != action_dim={action_dim}"
            )
        self._policy = policy
        self._policy.eval()
        self._action_dim = int(action_dim)

    @property
    def action_dim(self) -> int:
        return self._action_dim

    def reset(self, *, seed: int | None = None) -> None:
        del seed

    def act(self, observation: MyoArmState) -> np.ndarray:
        if not isinstance(observation, MyoArmState):
            raise ValueError(
                f"observation must be MyoArmState, got {type(observation).__name__}"
            )
        x = observation.flatten().astype(np.float32)
        with torch.no_grad():
            u = self._policy(torch.from_numpy(x).unsqueeze(0)).squeeze(0).numpy()
        return np.clip(u, _LO, _HI).astype(_DTYPE, copy=False)

    def __repr__(self) -> str:
        return (
            f"BCController(action_dim={self._action_dim}, "
            f"state_dim={self._policy.state_dim})"
        )
