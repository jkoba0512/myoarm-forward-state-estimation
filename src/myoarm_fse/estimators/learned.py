"""Stage A learned gain: condition-level supervised K predictor.

Stage A (this module) consumes only ``(controller, noise_sigma,
delay_steps)`` — the *condition* features that
``best_by_condition.csv`` is keyed by — and outputs a scalar K. Per
the Phase 3.2 design decisions, state-dependent inputs (innovation
norm, prediction uncertainty proxy, etc.) and timestep-level adaptation
are deferred to Stage B; here the K is computed once at estimator
construction and reused throughout the episode, so the per-step
behavior reduces to ``FixedGainKalmanEstimator`` and we delegate to it
via composition.

Module layout:

```text
- _encode_features            : (controller, sigma, delay) → (8,) np.float32
- GainPredictor               : nn.Module mapping features → scalar K ∈ [0, 1]
- LearnedGainKalmanEstimator  : Estimator Protocol implementation, composes
                                FixedGainKalmanEstimator with the inferred K
- LearnedGainTrainConfig      : training hyperparameters dataclass
- load_oracle_table           : best_by_condition.csv → list[dict]
- _make_cv_folds              : 4 supported strategies (loo / noise / delay /
                                controller)
- train_gain_predictor        : runs CV + final-model training on full N=36
- save_learned_gain_model     : state_dict + config.json + metrics.json + info.json
- load_learned_gain_model     : inverse of the above
- make_learned_gain_model_id  : UTC timestamp id (matches Step 5 / Step 8 format)
```
"""

from __future__ import annotations

import csv
import json
import random as py_random
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn

from myoarm_fse.envs.state import StateSpec
from myoarm_fse.estimators.fixed_kalman import FixedGainKalmanEstimator
from myoarm_fse.models.mlp import ForwardMLP


# Default categorical / numeric domains. Phase 3.3-min sweep used these.
_DEFAULT_CONTROLLER_NAMES: tuple[str, ...] = ("random", "lowamp", "hold")
_DEFAULT_SIGMA_FIELD_ORDER: tuple[str, ...] = (
    "qpos",
    "qvel",
    "tip_pos",
    "reach_err",
)
_DEFAULT_DELAY_MAX: int = 6


# --- feature encoder ---


def _encode_features(
    *,
    controller: str,
    noise_sigma: dict[str, float],
    delay_steps: int,
    controller_names: tuple[str, ...] = _DEFAULT_CONTROLLER_NAMES,
    sigma_field_order: tuple[str, ...] = _DEFAULT_SIGMA_FIELD_ORDER,
    delay_max: int = _DEFAULT_DELAY_MAX,
) -> np.ndarray:
    """Convert (controller, noise_sigma, delay_steps) to the 8-dim feature vec.

    Layout: ``[*controller_one_hot, *sigma_vec_in_order, delay/delay_max]``.
    Unknown controllers leave the one-hot all-zero. Sigma values default
    to 0 if a field is missing from ``noise_sigma``.
    """
    if delay_max <= 0:
        raise ValueError(f"delay_max must be > 0, got {delay_max}")
    controller_oh = np.zeros(len(controller_names), dtype=np.float32)
    if controller in controller_names:
        controller_oh[controller_names.index(controller)] = 1.0
    sigma_vec = np.array(
        [float(noise_sigma.get(name, 0.0)) for name in sigma_field_order],
        dtype=np.float32,
    )
    delay_norm = np.array([float(delay_steps) / float(delay_max)], dtype=np.float32)
    return np.concatenate([controller_oh, sigma_vec, delay_norm])


def feature_dim(
    n_controllers: int = len(_DEFAULT_CONTROLLER_NAMES),
    n_sigma_fields: int = len(_DEFAULT_SIGMA_FIELD_ORDER),
) -> int:
    """Return the encoder output dimension for the given category sizes."""
    return n_controllers + n_sigma_fields + 1


# --- model ---


class GainPredictor(nn.Module):
    """Small MLP mapping condition features to a scalar K ∈ [0, 1]."""

    def __init__(
        self,
        n_controllers: int = len(_DEFAULT_CONTROLLER_NAMES),
        n_sigma_fields: int = len(_DEFAULT_SIGMA_FIELD_ORDER),
        hidden_dims: Sequence[int] = (32, 32),
    ) -> None:
        super().__init__()
        if isinstance(n_controllers, bool) or not isinstance(n_controllers, int) or n_controllers <= 0:
            raise ValueError(
                f"n_controllers must be a positive int, got {n_controllers!r}"
            )
        if (
            isinstance(n_sigma_fields, bool)
            or not isinstance(n_sigma_fields, int)
            or n_sigma_fields <= 0
        ):
            raise ValueError(
                f"n_sigma_fields must be a positive int, got {n_sigma_fields!r}"
            )
        for h in hidden_dims:
            if isinstance(h, bool) or not isinstance(h, int) or h <= 0:
                raise ValueError(
                    f"each hidden_dim must be a positive int, got {h!r}"
                )
        in_dim = n_controllers + n_sigma_fields + 1
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
        self.n_controllers = int(n_controllers)
        self.n_sigma_fields = int(n_sigma_fields)
        self.hidden_dims = tuple(int(h) for h in hidden_dims)

    @property
    def in_dim(self) -> int:
        return self.n_controllers + self.n_sigma_fields + 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.in_dim:
            raise ValueError(
                f"x.shape[-1] must be {self.in_dim}, got {x.shape[-1]}"
            )
        logits = self.net(x).squeeze(-1)
        return torch.sigmoid(logits)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# --- estimator ---


class LearnedGainKalmanEstimator:
    """Kalman-like estimator using a context-derived scalar K (Stage A).

    The K is computed once at construction by running ``gain_predictor``
    on the encoded ``(controller, noise_sigma, delay_steps)`` feature
    vector. Per-step behavior is delegated to a composed
    ``FixedGainKalmanEstimator`` so delay handling, cold start and
    forward roll match Phase 3.1 exactly.
    """

    def __init__(
        self,
        forward_model: ForwardMLP,
        gain_predictor: GainPredictor,
        state_spec: StateSpec,
        *,
        delay_steps: int = 0,
        controller_name: str,
        noise_sigma: dict[str, float],
        controller_names: tuple[str, ...] = _DEFAULT_CONTROLLER_NAMES,
        sigma_field_order: tuple[str, ...] = _DEFAULT_SIGMA_FIELD_ORDER,
        delay_max: int = _DEFAULT_DELAY_MAX,
    ) -> None:
        if not isinstance(gain_predictor, GainPredictor):
            raise TypeError(
                f"gain_predictor must be GainPredictor, "
                f"got {type(gain_predictor).__name__}"
            )
        if controller_name not in controller_names:
            raise ValueError(
                f"controller_name {controller_name!r} not in known "
                f"controller_names {controller_names}"
            )
        features = _encode_features(
            controller=controller_name,
            noise_sigma=noise_sigma,
            delay_steps=delay_steps,
            controller_names=controller_names,
            sigma_field_order=sigma_field_order,
            delay_max=delay_max,
        )
        gain_predictor.eval()
        with torch.no_grad():
            tensor_in = torch.from_numpy(features).unsqueeze(0)
            k_scalar = float(gain_predictor(tensor_in).item())
        # Defensive clamp (sigmoid output is already in [0,1] but float32
        # round-off near the boundary can land just outside).
        k_scalar = float(np.clip(k_scalar, 0.0, 1.0))
        self._learned_k: float = k_scalar
        self._controller_name = str(controller_name)
        self._noise_sigma = dict(noise_sigma)
        self._inner = FixedGainKalmanEstimator(
            forward_model=forward_model,
            gain=k_scalar,
            state_spec=state_spec,
            delay_steps=delay_steps,
        )

    @property
    def state_dim(self) -> int:
        return self._inner.state_dim

    @property
    def action_dim(self) -> int:
        return self._inner.action_dim

    @property
    def delay_steps(self) -> int:
        return self._inner.delay_steps

    @property
    def gain_vec(self) -> np.ndarray:
        return self._inner.gain_vec

    @property
    def learned_k(self) -> float:
        """Scalar K inferred at construction (read-only)."""
        return self._learned_k

    @property
    def controller_name(self) -> str:
        return self._controller_name

    @property
    def noise_sigma(self) -> dict[str, float]:
        return dict(self._noise_sigma)

    def reset(self, initial_state: np.ndarray) -> None:
        self._inner.reset(initial_state)

    def step(self, y_obs: np.ndarray, u: np.ndarray) -> np.ndarray:
        return self._inner.step(y_obs, u)


# --- training config ---


@dataclass(frozen=True)
class LearnedGainTrainConfig:
    """Hyperparameters for ``train_gain_predictor``."""

    optimizer: str = "adam"
    lr: float = 1e-3
    weight_decay: float = 1e-3
    batch_size: int = 36
    epochs: int = 500
    seed: int = 0
    cv_strategies: tuple[str, ...] = ("loo", "noise", "delay", "controller")

    def __post_init__(self) -> None:
        if self.optimizer not in ("adam", "adamw"):
            raise ValueError(
                f"optimizer must be 'adam' or 'adamw', got {self.optimizer!r}"
            )
        for name in ("batch_size", "epochs"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, int) or v <= 0:
                raise ValueError(f"{name} must be a positive int, got {v!r}")
        for name in ("lr", "weight_decay"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or v < 0.0 or not np.isfinite(v):
                raise ValueError(
                    f"{name} must be a non-negative finite number, got {v!r}"
                )
        valid_strategies = {"loo", "noise", "delay", "controller"}
        for s in self.cv_strategies:
            if s not in valid_strategies:
                raise ValueError(
                    f"unknown cv_strategy {s!r}; valid: {sorted(valid_strategies)}"
                )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LearnedGainTrainConfig:
        valid = {f.name for f in fields(cls)}
        unknown = set(data.keys()) - valid
        if unknown:
            raise ValueError(
                f"unknown LearnedGainTrainConfig keys: {sorted(unknown)}"
            )
        # cv_strategies may come as list from YAML; coerce to tuple.
        if "cv_strategies" in data and isinstance(data["cv_strategies"], list):
            data = dict(data)
            data["cv_strategies"] = tuple(data["cv_strategies"])
        return cls(**data)


# --- oracle table loader ---


def load_oracle_table(path: str | Path) -> list[dict[str, Any]]:
    """Read ``best_by_condition.csv`` into a list of dicts.

    Expected columns include at least ``controller``, ``noise_condition``,
    ``delay_steps``, ``gain`` (the oracle K). Numeric fields are coerced
    to float / int; the rest are returned as strings. Sigma vectors are
    NOT in the CSV; the caller can attach a noise sigma lookup via
    ``noise_conditions`` from the eval config.
    """
    path = Path(path)
    rows: list[dict[str, Any]] = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for raw in reader:
            row: dict[str, Any] = dict(raw)
            for k in ("delay_steps", "n_episodes"):
                if k in row and row[k] != "":
                    row[k] = int(float(row[k]))
            for k in (
                "gain",
                "tip_estimation_error_mean",
                "tip_estimation_error_final",
                "tip_estimation_error_std",
                "state_mse_mean",
                "mse_qpos_mean",
                "mse_qvel_mean",
                "mse_act_mean",
                "mse_tip_pos_mean",
                "mse_target_pos_mean",
                "mse_reach_err_mean",
            ):
                if k in row and row[k] != "":
                    try:
                        row[k] = float(row[k])
                    except ValueError:
                        pass
            rows.append(row)
    if not rows:
        raise ValueError(f"oracle table at {path} contains no rows")
    return rows


# --- CV folds ---


def _make_cv_folds(
    rows: list[dict[str, Any]],
    strategy: str,
) -> list[tuple[list[int], list[int]]]:
    """Return ``[(train_idx, test_idx), ...]`` for the given CV strategy.

    Strategies:

    - ``loo``: one fold per row, test = single row, train = the rest.
    - ``noise`` / ``delay`` / ``controller``: hold out all rows whose
      respective key matches each unique value, train on the others.
    """
    if strategy == "loo":
        folds: list[tuple[list[int], list[int]]] = []
        for i in range(len(rows)):
            train_idx = [j for j in range(len(rows)) if j != i]
            test_idx = [i]
            folds.append((train_idx, test_idx))
        return folds
    if strategy == "noise":
        key = "noise_condition"
    elif strategy == "delay":
        key = "delay_steps"
    elif strategy == "controller":
        key = "controller"
    else:
        raise ValueError(
            f"unknown cv strategy {strategy!r}; "
            "valid: 'loo' | 'noise' | 'delay' | 'controller'"
        )
    # Group rows by key.
    by_value: dict[Any, list[int]] = {}
    for i, row in enumerate(rows):
        by_value.setdefault(row[key], []).append(i)
    # Sort keys deterministically for reproducible fold ordering.
    sorted_values = sorted(
        by_value.keys(),
        key=lambda v: (str(type(v).__name__), str(v)),
    )
    folds = []
    for v in sorted_values:
        test_idx = list(by_value[v])
        train_idx = [j for j in range(len(rows)) if j not in set(test_idx)]
        folds.append((train_idx, test_idx))
    return folds


# --- training ---


def _seed_everything(master_seed: int) -> dict[str, int]:
    """Reproduce numpy / torch / random seeding via SeedSequence.spawn."""
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
    py_random.seed(seeds["model_init"])
    return seeds


def _build_optimizer(
    model: nn.Module, config: LearnedGainTrainConfig
) -> torch.optim.Optimizer:
    if config.optimizer == "adam":
        return torch.optim.Adam(
            model.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )
    return torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )


def _train_one_model(
    rows: list[dict[str, Any]],
    train_idx: list[int],
    feature_matrix: np.ndarray,
    target_vector: np.ndarray,
    config: LearnedGainTrainConfig,
    *,
    n_controllers: int,
    n_sigma_fields: int,
    hidden_dims: Sequence[int],
) -> tuple[GainPredictor, list[float]]:
    """Train a fresh GainPredictor on ``train_idx`` rows. Returns model + loss curve."""
    model = GainPredictor(
        n_controllers=n_controllers,
        n_sigma_fields=n_sigma_fields,
        hidden_dims=hidden_dims,
    )
    optimizer = _build_optimizer(model, config)
    train_x = torch.from_numpy(feature_matrix[train_idx])
    train_y = torch.from_numpy(target_vector[train_idx])
    loss_fn = nn.MSELoss()
    history: list[float] = []
    model.train()
    for _ in range(config.epochs):
        optimizer.zero_grad(set_to_none=True)
        pred = model(train_x)
        loss = loss_fn(pred, train_y)
        loss.backward()
        optimizer.step()
        history.append(float(loss.item()))
    model.eval()
    return model, history


def _predict_K(model: GainPredictor, features: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(features)).cpu().numpy()


def train_gain_predictor(
    rows: list[dict[str, Any]],
    *,
    config: LearnedGainTrainConfig,
    noise_conditions: dict[str, dict[str, float]],
    controller_names: tuple[str, ...] = _DEFAULT_CONTROLLER_NAMES,
    sigma_field_order: tuple[str, ...] = _DEFAULT_SIGMA_FIELD_ORDER,
    delay_max: int = _DEFAULT_DELAY_MAX,
    hidden_dims: Sequence[int] = (32, 32),
) -> tuple[GainPredictor, dict[str, Any]]:
    """Run CV + train final model on the full oracle table.

    Returns ``(final_model, metrics)`` where metrics contains per-strategy
    CV results (per-fold abs error stats and aggregate mean / median /
    max / std) and the final-training loss curve.
    """
    if not rows:
        raise ValueError("rows must be non-empty")
    if not noise_conditions:
        raise ValueError("noise_conditions must be a non-empty mapping")

    # Build feature matrix and target vector aligned to row order.
    n = len(rows)
    feature_matrix = np.empty(
        (n, len(controller_names) + len(sigma_field_order) + 1),
        dtype=np.float32,
    )
    target_vector = np.empty(n, dtype=np.float32)
    for i, row in enumerate(rows):
        cond = row["noise_condition"]
        if cond not in noise_conditions:
            raise ValueError(
                f"row[{i}].noise_condition={cond!r} not in noise_conditions "
                f"keys {sorted(noise_conditions.keys())}"
            )
        sigma_dict = noise_conditions[cond]
        feature_matrix[i] = _encode_features(
            controller=row["controller"],
            noise_sigma=sigma_dict,
            delay_steps=int(row["delay_steps"]),
            controller_names=controller_names,
            sigma_field_order=sigma_field_order,
            delay_max=delay_max,
        )
        target_vector[i] = float(row["gain"])

    seeds = _seed_everything(config.seed)

    cv_results: dict[str, dict[str, Any]] = {}
    for strategy in config.cv_strategies:
        folds = _make_cv_folds(rows, strategy)
        per_fold = []
        all_abs_errors: list[float] = []
        for fold_idx, (train_idx, test_idx) in enumerate(folds):
            model, _hist = _train_one_model(
                rows, train_idx, feature_matrix, target_vector, config,
                n_controllers=len(controller_names),
                n_sigma_fields=len(sigma_field_order),
                hidden_dims=hidden_dims,
            )
            preds = _predict_K(model, feature_matrix[test_idx])
            errs = np.abs(preds - target_vector[test_idx])
            per_fold.append({
                "fold": fold_idx,
                "n_test": int(errs.shape[0]),
                "abs_error_mean": float(np.mean(errs)),
                "abs_error_max": float(np.max(errs)),
            })
            all_abs_errors.extend(errs.tolist())
        cv_results[strategy] = {
            "n_folds": len(folds),
            "abs_error_mean": float(np.mean(all_abs_errors)),
            "abs_error_median": float(np.median(all_abs_errors)),
            "abs_error_std": float(np.std(all_abs_errors)),
            "abs_error_max": float(np.max(all_abs_errors)),
            "per_fold": per_fold,
        }

    # Final model: train on full N=36.
    final_model, final_history = _train_one_model(
        rows,
        train_idx=list(range(n)),
        feature_matrix=feature_matrix,
        target_vector=target_vector,
        config=config,
        n_controllers=len(controller_names),
        n_sigma_fields=len(sigma_field_order),
        hidden_dims=hidden_dims,
    )

    final_preds = _predict_K(final_model, feature_matrix)
    final_abs_errors = np.abs(final_preds - target_vector)
    metrics: dict[str, Any] = {
        "n_oracle_rows": n,
        "feature_dim": int(feature_matrix.shape[1]),
        "seeds": seeds,
        "cv_results": cv_results,
        "final_train_loss_history": final_history,
        "final_train_abs_error_mean": float(np.mean(final_abs_errors)),
        "final_train_abs_error_max": float(np.max(final_abs_errors)),
    }
    return final_model, metrics


# --- save / load ---


def make_learned_gain_model_id(now: datetime | None = None) -> str:
    """UTC timestamp matching the Step 5 / Step 8 format."""
    t = now if now is not None else datetime.now(timezone.utc)
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    return t.astimezone(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


def save_learned_gain_model(
    model: GainPredictor,
    config: dict[str, Any],
    metrics: dict[str, Any],
    *,
    path: str | Path,
    info: dict[str, Any] | None = None,
) -> None:
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


def load_learned_gain_model(
    path: str | Path,
) -> tuple[GainPredictor, dict[str, Any], dict[str, Any]]:
    p = Path(path)
    config = json.loads((p / "config.json").read_text())
    arch = config.get("architecture", {})
    n_controllers = int(arch.get("n_controllers", len(_DEFAULT_CONTROLLER_NAMES)))
    n_sigma_fields = int(arch.get("n_sigma_fields", len(_DEFAULT_SIGMA_FIELD_ORDER)))
    hidden_dims = tuple(int(h) for h in arch.get("hidden_dims", (32, 32)))
    model = GainPredictor(
        n_controllers=n_controllers,
        n_sigma_fields=n_sigma_fields,
        hidden_dims=hidden_dims,
    )
    state_dict = torch.load(p / "model.pt", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    metrics = json.loads((p / "metrics.json").read_text())
    return model, config, metrics
