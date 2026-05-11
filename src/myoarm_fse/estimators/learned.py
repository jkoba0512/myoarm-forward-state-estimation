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

# Stage B per-step state features: per-field L2 norms of the innovation
# vector. Same field set as the sigma field order — keeps the input
# compact (4-dim) and lines each component up with the corresponding
# noise scale in the condition features so the model can learn signal-
# to-noise interactions.
_DEFAULT_STATE_FEATURE_FIELDS: tuple[str, ...] = (
    "qpos",
    "qvel",
    "tip_pos",
    "reach_err",
)


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


def _encode_state_features(
    innovation_flat: np.ndarray,
    state_spec: StateSpec,
    *,
    state_feature_fields: tuple[str, ...] = _DEFAULT_STATE_FEATURE_FIELDS,
) -> np.ndarray:
    """Return per-field L2 norms of the innovation vector (Stage B)."""
    if innovation_flat.shape != (state_spec.dim,):
        raise ValueError(
            f"innovation_flat must have shape ({state_spec.dim},), "
            f"got {innovation_flat.shape}"
        )
    layout = state_spec.layout()
    unknown = set(state_feature_fields) - set(layout.keys())
    if unknown:
        raise ValueError(
            f"unknown state_feature_fields: {sorted(unknown)}; "
            f"valid: {sorted(layout.keys())}"
        )
    out = np.empty(len(state_feature_fields), dtype=np.float32)
    for i, name in enumerate(state_feature_fields):
        out[i] = float(np.linalg.norm(innovation_flat[layout[name]]))
    return out


def state_aware_feature_dim(
    n_controllers: int = len(_DEFAULT_CONTROLLER_NAMES),
    n_sigma_fields: int = len(_DEFAULT_SIGMA_FIELD_ORDER),
    n_state_features: int = len(_DEFAULT_STATE_FEATURE_FIELDS),
) -> int:
    """Return the Stage B encoder output dimension."""
    return n_controllers + n_sigma_fields + 1 + n_state_features


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


class StateAwareGainPredictor(nn.Module):
    """Stage B: MLP mapping condition + per-step state features to K ∈ [0, 1].

    Inputs concatenate the Stage A condition vector
    (controller one-hot + sigma vec + delay/delay_max, length ``8`` by
    default) with the Stage B state feature vector (per-field innovation
    L2 norms, length ``4`` by default), for a total ``in_dim`` of
    ``n_controllers + n_sigma_fields + 1 + n_state_features``.
    """

    def __init__(
        self,
        n_controllers: int = len(_DEFAULT_CONTROLLER_NAMES),
        n_sigma_fields: int = len(_DEFAULT_SIGMA_FIELD_ORDER),
        n_state_features: int = len(_DEFAULT_STATE_FEATURE_FIELDS),
        hidden_dims: Sequence[int] = (64, 64),
    ) -> None:
        super().__init__()
        for name, var in (
            ("n_controllers", n_controllers),
            ("n_sigma_fields", n_sigma_fields),
            ("n_state_features", n_state_features),
        ):
            if isinstance(var, bool) or not isinstance(var, int) or var <= 0:
                raise ValueError(f"{name} must be a positive int, got {var!r}")
        for h in hidden_dims:
            if isinstance(h, bool) or not isinstance(h, int) or h <= 0:
                raise ValueError(
                    f"each hidden_dim must be a positive int, got {h!r}"
                )
        in_dim = n_controllers + n_sigma_fields + 1 + n_state_features
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
        self.n_state_features = int(n_state_features)
        self.hidden_dims = tuple(int(h) for h in hidden_dims)

    @property
    def in_dim(self) -> int:
        return (
            self.n_controllers
            + self.n_sigma_fields
            + 1
            + self.n_state_features
        )

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


class StateAwareLearnedGainKalmanEstimator:
    """Stage B: Kalman-like estimator with per-step K from a state-aware predictor.

    On every ``step()`` the predictor is evaluated against the
    concatenation of the (constant) condition feature vector and a
    per-step state feature vector derived from the current innovation.
    Internally we mirror :class:`FixedGainKalmanEstimator` so delay
    handling and cold start match Phase 3.1 exactly — the only change
    is that K is recomputed per correction rather than fixed at
    construction.
    """

    def __init__(
        self,
        forward_model: ForwardMLP,
        gain_predictor: StateAwareGainPredictor,
        state_spec: StateSpec,
        *,
        delay_steps: int = 0,
        controller_name: str,
        noise_sigma: dict[str, float],
        controller_names: tuple[str, ...] = _DEFAULT_CONTROLLER_NAMES,
        sigma_field_order: tuple[str, ...] = _DEFAULT_SIGMA_FIELD_ORDER,
        state_feature_fields: tuple[str, ...] = _DEFAULT_STATE_FEATURE_FIELDS,
        delay_max: int = _DEFAULT_DELAY_MAX,
    ) -> None:
        if not isinstance(gain_predictor, StateAwareGainPredictor):
            raise TypeError(
                f"gain_predictor must be StateAwareGainPredictor, "
                f"got {type(gain_predictor).__name__}"
            )
        if not isinstance(forward_model, ForwardMLP):
            raise TypeError(
                f"forward_model must be ForwardMLP, "
                f"got {type(forward_model).__name__}"
            )
        if not isinstance(state_spec, StateSpec):
            raise TypeError(
                f"state_spec must be StateSpec, got {type(state_spec).__name__}"
            )
        if forward_model.state_dim != state_spec.dim:
            raise ValueError(
                f"forward_model.state_dim={forward_model.state_dim} does not "
                f"match state_spec.dim={state_spec.dim}"
            )
        if isinstance(delay_steps, bool) or not isinstance(delay_steps, int):
            raise ValueError(
                f"delay_steps must be a non-negative int, got {delay_steps!r}"
            )
        if delay_steps < 0:
            raise ValueError(f"delay_steps must be >= 0, got {delay_steps}")
        if controller_name not in controller_names:
            raise ValueError(
                f"controller_name {controller_name!r} not in known "
                f"controller_names {controller_names}"
            )

        cond_features = _encode_features(
            controller=controller_name,
            noise_sigma=noise_sigma,
            delay_steps=delay_steps,
            controller_names=controller_names,
            sigma_field_order=sigma_field_order,
            delay_max=delay_max,
        )
        gain_predictor.eval()

        self._forward_model = forward_model
        self._state_spec = state_spec
        self._delay_steps = int(delay_steps)
        self._gain_predictor = gain_predictor
        self._cond_features = torch.from_numpy(cond_features)
        self._state_feature_fields = tuple(state_feature_fields)
        self._controller_name = str(controller_name)
        self._noise_sigma = dict(noise_sigma)
        # Per-step K log: appended on every correction (skipped during
        # delay>0 cold start). Tests / eval can read it to verify the
        # predictor is varying K through the episode.
        self._k_history: list[float] = []

        # Same buffer layout as FixedGainKalmanEstimator.
        from collections import deque  # local import to avoid surprises
        self._deque = deque
        self._x_buffer: Any = None
        self._u_buffer: Any = None

    @property
    def state_dim(self) -> int:
        return int(self._state_spec.dim)

    @property
    def action_dim(self) -> int:
        return int(self._forward_model.action_dim)

    @property
    def delay_steps(self) -> int:
        return self._delay_steps

    @property
    def controller_name(self) -> str:
        return self._controller_name

    @property
    def noise_sigma(self) -> dict[str, float]:
        return dict(self._noise_sigma)

    @property
    def k_history(self) -> list[float]:
        """List of K values used at each correction step (read-only copy)."""
        return list(self._k_history)

    def reset(self, initial_state: np.ndarray) -> None:
        arr = np.asarray(initial_state, dtype=np.float32)
        if arr.shape != (self.state_dim,):
            raise ValueError(
                f"initial_state must have shape ({self.state_dim},), "
                f"got {arr.shape}"
            )
        if not np.isfinite(arr).all():
            raise ValueError("initial_state contains non-finite values")
        self._x_buffer = self._deque(
            [arr.copy() for _ in range(self._delay_steps + 1)],
            maxlen=self._delay_steps + 1,
        )
        self._u_buffer = self._deque(maxlen=self._delay_steps)
        self._k_history = []

    def step(self, y_obs: np.ndarray, u: np.ndarray) -> np.ndarray:
        if self._x_buffer is None or self._u_buffer is None:
            raise RuntimeError(
                "StateAwareLearnedGainKalmanEstimator.step called before reset; "
                "call reset(initial_state) at the start of each episode"
            )
        y_arr = np.asarray(y_obs, dtype=np.float32)
        if y_arr.shape != (self.state_dim,):
            raise ValueError(
                f"y_obs must have shape ({self.state_dim},), got {y_arr.shape}"
            )
        if not np.isfinite(y_arr).all():
            raise ValueError("y_obs contains non-finite values")
        u_arr = np.asarray(u, dtype=np.float32)
        if u_arr.shape != (self.action_dim,):
            raise ValueError(
                f"u must have shape ({self.action_dim},), got {u_arr.shape}"
            )

        past_actions = list(self._u_buffer)
        current = self._x_buffer[-1]
        x_pred_next = current + self._predict_delta(current, u_arr)

        if self._delay_steps == 0:
            innovation = y_arr - x_pred_next
            k_scalar = self._compute_k(innovation)
            corrected = x_pred_next + k_scalar * innovation
            self._x_buffer[-1] = corrected.astype(np.float32, copy=False)
            self._k_history.append(k_scalar)
            return corrected.astype(np.float32, copy=True)

        if len(past_actions) < self._delay_steps:
            # Cold start: propagate prediction without correction. K log
            # left untouched to keep history aligned with corrections.
            self._x_buffer.append(x_pred_next.astype(np.float32, copy=False))
            self._u_buffer.append(u_arr.copy())
            return x_pred_next.astype(np.float32, copy=True)

        x_at_past = self._x_buffer[0]
        innovation = y_arr - x_at_past
        k_scalar = self._compute_k(innovation)
        corrected_past = x_at_past + k_scalar * innovation
        x = corrected_past
        for past_u in past_actions:
            x = x + self._predict_delta(x, past_u)
        self._x_buffer.append(x.astype(np.float32, copy=False))
        self._u_buffer.append(u_arr.copy())
        self._k_history.append(k_scalar)
        return x.astype(np.float32, copy=True)

    def _compute_k(self, innovation_flat: np.ndarray) -> float:
        state_feats = _encode_state_features(
            innovation_flat,
            state_spec=self._state_spec,
            state_feature_fields=self._state_feature_fields,
        )
        x_in = torch.cat(
            [self._cond_features, torch.from_numpy(state_feats)]
        ).unsqueeze(0)
        with torch.no_grad():
            k = float(self._gain_predictor(x_in).item())
        return float(np.clip(k, 0.0, 1.0))

    def _predict_delta(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        x_t = torch.from_numpy(x).unsqueeze(0)
        u_t = torch.from_numpy(u).unsqueeze(0)
        with torch.no_grad():
            dx = self._forward_model(x_t, u_t).squeeze(0).numpy()
        return dx.astype(np.float32, copy=False)


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


# --- Stage B training ---


@dataclass(frozen=True)
class StateAwareTrainConfig:
    """Hyperparameters for ``train_state_aware_gain_predictor``."""

    optimizer: str = "adam"
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 1024
    epochs: int = 30
    seed: int = 0
    cv_strategies: tuple[str, ...] = ("condition", "noise", "delay", "controller")

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
            if (
                isinstance(v, bool)
                or not isinstance(v, (int, float))
                or v < 0.0
                or not np.isfinite(v)
            ):
                raise ValueError(
                    f"{name} must be a non-negative finite number, got {v!r}"
                )
        valid_strategies = {"condition", "noise", "delay", "controller"}
        for s in self.cv_strategies:
            if s not in valid_strategies:
                raise ValueError(
                    f"unknown cv_strategy {s!r}; valid: {sorted(valid_strategies)}"
                )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StateAwareTrainConfig":
        valid = {f.name for f in fields(cls)}
        unknown = set(data.keys()) - valid
        if unknown:
            raise ValueError(
                f"unknown StateAwareTrainConfig keys: {sorted(unknown)}"
            )
        if "cv_strategies" in data and isinstance(data["cv_strategies"], list):
            data = dict(data)
            data["cv_strategies"] = tuple(data["cv_strategies"])
        return cls(**data)


def _make_state_aware_cv_folds(
    groups: np.ndarray,
    strategy: str,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """One fold per unique value in ``groups``.

    ``groups`` is a 1-D int array of group ids (length = n_samples).
    For each unique id, the fold's test set is all samples with that id
    and the train set is everyone else.
    """
    unique = np.unique(groups)
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for v in unique:
        test_mask = groups == v
        test_idx = np.where(test_mask)[0]
        train_idx = np.where(~test_mask)[0]
        if train_idx.size == 0 or test_idx.size == 0:
            continue
        folds.append((train_idx, test_idx))
    return folds


def _train_state_aware_one_model(
    train_x: np.ndarray,
    train_y: np.ndarray,
    config: StateAwareTrainConfig,
    *,
    n_controllers: int,
    n_sigma_fields: int,
    n_state_features: int,
    hidden_dims: Sequence[int],
    seed: int,
) -> tuple[StateAwareGainPredictor, list[float]]:
    """Train a fresh StateAwareGainPredictor with shuffled minibatch SGD."""
    torch.manual_seed(seed)
    model = StateAwareGainPredictor(
        n_controllers=n_controllers,
        n_sigma_fields=n_sigma_fields,
        n_state_features=n_state_features,
        hidden_dims=hidden_dims,
    )
    if config.optimizer == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )
    loss_fn = nn.MSELoss()
    n = train_x.shape[0]
    rng = np.random.default_rng(seed)
    history: list[float] = []
    x_t = torch.from_numpy(train_x)
    y_t = torch.from_numpy(train_y)
    model.train()
    for _epoch in range(config.epochs):
        perm = rng.permutation(n)
        epoch_loss_sum = 0.0
        epoch_count = 0
        for start in range(0, n, config.batch_size):
            idx = perm[start : start + config.batch_size]
            xb = x_t[idx]
            yb = y_t[idx]
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss_sum += float(loss.item()) * xb.shape[0]
            epoch_count += xb.shape[0]
        history.append(epoch_loss_sum / max(epoch_count, 1))
    model.eval()
    return model, history


def _predict_state_aware(
    model: StateAwareGainPredictor, x: np.ndarray
) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(x)).cpu().numpy()


def train_state_aware_gain_predictor(
    cond_features: np.ndarray,
    state_features: np.ndarray,
    k_target: np.ndarray,
    groups: dict[str, np.ndarray],
    *,
    config: StateAwareTrainConfig,
    n_controllers: int = len(_DEFAULT_CONTROLLER_NAMES),
    n_sigma_fields: int = len(_DEFAULT_SIGMA_FIELD_ORDER),
    n_state_features: int = len(_DEFAULT_STATE_FEATURE_FIELDS),
    hidden_dims: Sequence[int] = (64, 64),
) -> tuple[StateAwareGainPredictor, dict[str, Any]]:
    """Train Stage B predictor on per-step samples with grouped CV.

    Inputs ``cond_features`` (N, ``n_controllers + n_sigma_fields + 1``)
    and ``state_features`` (N, ``n_state_features``) are concatenated
    horizontally to form the model input. ``groups`` maps CV strategy
    name to a length-N int array of group ids; only strategies listed
    in ``config.cv_strategies`` are evaluated.
    """
    if cond_features.ndim != 2 or state_features.ndim != 2:
        raise ValueError("cond_features and state_features must be 2-D")
    if cond_features.shape[0] != state_features.shape[0]:
        raise ValueError("cond_features and state_features row counts must match")
    n = cond_features.shape[0]
    if n == 0:
        raise ValueError("samples must be non-empty")
    if k_target.shape != (n,):
        raise ValueError(f"k_target must have shape ({n},), got {k_target.shape}")

    feature_matrix = np.concatenate(
        [cond_features.astype(np.float32), state_features.astype(np.float32)],
        axis=1,
    )
    target_vec = k_target.astype(np.float32)

    seeds = _seed_everything(config.seed)

    cv_results: dict[str, dict[str, Any]] = {}
    for strategy in config.cv_strategies:
        if strategy not in groups:
            raise ValueError(
                f"groups missing key {strategy!r} required by cv_strategies"
            )
        folds = _make_state_aware_cv_folds(groups[strategy], strategy)
        if not folds:
            cv_results[strategy] = {"n_folds": 0}
            continue
        per_fold: list[dict[str, Any]] = []
        all_abs_errors: list[float] = []
        for fold_idx, (train_idx, test_idx) in enumerate(folds):
            model, _hist = _train_state_aware_one_model(
                feature_matrix[train_idx],
                target_vec[train_idx],
                config,
                n_controllers=n_controllers,
                n_sigma_fields=n_sigma_fields,
                n_state_features=n_state_features,
                hidden_dims=hidden_dims,
                seed=int(seeds["model_init"]) + fold_idx,
            )
            preds = _predict_state_aware(model, feature_matrix[test_idx])
            errs = np.abs(preds - target_vec[test_idx])
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

    final_model, final_history = _train_state_aware_one_model(
        feature_matrix,
        target_vec,
        config,
        n_controllers=n_controllers,
        n_sigma_fields=n_sigma_fields,
        n_state_features=n_state_features,
        hidden_dims=hidden_dims,
        seed=int(seeds["model_init"]),
    )
    final_preds = _predict_state_aware(final_model, feature_matrix)
    final_abs_errors = np.abs(final_preds - target_vec)

    metrics: dict[str, Any] = {
        "n_samples": int(n),
        "feature_dim": int(feature_matrix.shape[1]),
        "seeds": seeds,
        "cv_results": cv_results,
        "final_train_loss_history": final_history,
        "final_train_abs_error_mean": float(np.mean(final_abs_errors)),
        "final_train_abs_error_max": float(np.max(final_abs_errors)),
        "target_k_mean": float(np.mean(target_vec)),
        "predicted_k_mean": float(np.mean(final_preds)),
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
    model: GainPredictor | StateAwareGainPredictor,
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
) -> tuple[GainPredictor | StateAwareGainPredictor, dict[str, Any], dict[str, Any]]:
    """Load a saved learned-gain model.

    Dispatches on ``config['architecture']['kind']``: ``GainPredictor``
    (Stage A, default) or ``StateAwareGainPredictor`` (Stage B). The
    returned model's class matches the kind saved at training time.
    """
    p = Path(path)
    config = json.loads((p / "config.json").read_text())
    arch = config.get("architecture", {})
    kind = arch.get("kind", "GainPredictor")
    n_controllers = int(arch.get("n_controllers", len(_DEFAULT_CONTROLLER_NAMES)))
    n_sigma_fields = int(arch.get("n_sigma_fields", len(_DEFAULT_SIGMA_FIELD_ORDER)))
    if kind == "GainPredictor":
        hidden_dims = tuple(int(h) for h in arch.get("hidden_dims", (32, 32)))
        model: GainPredictor | StateAwareGainPredictor = GainPredictor(
            n_controllers=n_controllers,
            n_sigma_fields=n_sigma_fields,
            hidden_dims=hidden_dims,
        )
    elif kind == "StateAwareGainPredictor":
        n_state_features = int(
            arch.get("n_state_features", len(_DEFAULT_STATE_FEATURE_FIELDS))
        )
        hidden_dims = tuple(int(h) for h in arch.get("hidden_dims", (64, 64)))
        model = StateAwareGainPredictor(
            n_controllers=n_controllers,
            n_sigma_fields=n_sigma_fields,
            n_state_features=n_state_features,
            hidden_dims=hidden_dims,
        )
    else:
        raise ValueError(
            f"unknown architecture.kind {kind!r}; "
            "expected 'GainPredictor' or 'StateAwareGainPredictor'"
        )
    state_dict = torch.load(p / "model.pt", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    metrics = json.loads((p / "metrics.json").read_text())
    return model, config, metrics
