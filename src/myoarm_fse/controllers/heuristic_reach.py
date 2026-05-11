"""Heuristic reach controller (Phase 2 MVP).

Maps the observation's ``reach_err = tip_pos - target_pos`` to a 34-dim
muscle excitation through a fixed linear projection ``W`` followed by a
shift + sigmoid:

```
u = sigmoid(logit_base + gain * W @ (-reach_err))
```

The negative sign makes the controller activate muscles that pull the
tip toward the target. ``W`` is sampled once from ``np.random.default_rng(W_seed)``
so the controller is deterministic across runs given the same seed. The
output is guaranteed in ``[0, 1]`` by the sigmoid, with values clipped
for numerical safety.

The controller is **not** meant to be a high-quality reaching policy —
it exists so closed-loop evaluation can compare different estimators by
feeding ``observation`` from the estimator output. As long as ``act()``
varies meaningfully with ``reach_err``, estimation quality differences
propagate to task-level metrics.
"""

from __future__ import annotations

import numpy as np

from myoarm_fse.envs.state import MyoArmState

_DTYPE: np.dtype = np.dtype(np.float32)
_LO: float = 0.0
_HI: float = 1.0
_CART_DIM: int = 3


def _coerce_W(
    W: object,
    action_dim: int,
    seed: int,
    scale: float = 1.0,
) -> np.ndarray:
    """Build (action_dim, 3) projection matrix from explicit array or seed."""
    if W is None:
        rng = np.random.default_rng(seed)
        # Standard-normal entries scaled by ``scale``. Negative entries
        # let muscles activate when -reach_err points in either
        # direction along a Cartesian axis — i.e., antagonist pairs.
        return rng.standard_normal(size=(action_dim, _CART_DIM)).astype(
            _DTYPE
        ) * float(scale)
    arr = np.asarray(W, dtype=_DTYPE)
    if arr.shape != (action_dim, _CART_DIM):
        raise ValueError(
            f"W must have shape ({action_dim}, {_CART_DIM}), got {arr.shape}"
        )
    if not np.isfinite(arr).all():
        raise ValueError("W contains non-finite values")
    return arr.copy()


class HeuristicReachController:
    """Sigmoid-of-linear controller driven by estimated reach error."""

    def __init__(
        self,
        action_dim: int,
        *,
        logit_base: float = 0.0,
        gain: float = 5.0,
        W: object = None,
        W_seed: int = 0,
        W_scale: float = 1.0,
    ) -> None:
        if isinstance(action_dim, bool):
            raise ValueError(
                f"action_dim must be a positive int, got bool: {action_dim!r}"
            )
        if not isinstance(action_dim, int):
            raise ValueError(
                f"action_dim must be a positive int, "
                f"got {type(action_dim).__name__}: {action_dim!r}"
            )
        if action_dim <= 0:
            raise ValueError(f"action_dim must be > 0, got {action_dim}")
        for name, val in (("logit_base", logit_base), ("gain", gain)):
            if isinstance(val, bool):
                raise ValueError(f"{name} must not be bool, got {val!r}")
            if not isinstance(val, (int, float, np.floating)):
                raise ValueError(
                    f"{name} must be numeric, got {type(val).__name__}"
                )
            if not np.isfinite(float(val)):
                raise ValueError(f"{name} must be finite, got {val!r}")
        if isinstance(W_seed, bool) or not isinstance(W_seed, int):
            raise ValueError(
                f"W_seed must be int, got {type(W_seed).__name__}: {W_seed!r}"
            )
        if (
            isinstance(W_scale, bool)
            or not isinstance(W_scale, (int, float, np.floating))
            or not np.isfinite(float(W_scale))
            or float(W_scale) <= 0.0
        ):
            raise ValueError(f"W_scale must be a positive finite number, got {W_scale!r}")
        self._action_dim: int = int(action_dim)
        self._logit_base: float = float(logit_base)
        self._gain: float = float(gain)
        self._W: np.ndarray = _coerce_W(W, action_dim, int(W_seed), float(W_scale))
        self._W_seed: int = int(W_seed)
        self._W_scale: float = float(W_scale)

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def logit_base(self) -> float:
        return self._logit_base

    @property
    def gain(self) -> float:
        return self._gain

    @property
    def W(self) -> np.ndarray:
        return self._W.copy()

    @property
    def W_seed(self) -> int:
        return self._W_seed

    def reset(self, *, seed: int | None = None) -> None:
        # The controller has no per-episode internal state. Accepting
        # ``seed`` keeps the Controller protocol uniform.
        del seed

    def act(self, observation: MyoArmState) -> np.ndarray:
        if not isinstance(observation, MyoArmState):
            raise ValueError(
                f"observation must be MyoArmState, got {type(observation).__name__}"
            )
        reach_err = np.asarray(observation.reach_err, dtype=_DTYPE)
        if reach_err.shape != (_CART_DIM,):
            raise ValueError(
                f"observation.reach_err must have shape ({_CART_DIM},), "
                f"got {reach_err.shape}"
            )
        # Direction "toward target" is -reach_err.
        drive = self._W @ (-reach_err)
        logits = self._logit_base + self._gain * drive
        u = 1.0 / (1.0 + np.exp(-logits))
        return np.clip(u, _LO, _HI).astype(_DTYPE, copy=False)

    def __repr__(self) -> str:
        return (
            f"HeuristicReachController(action_dim={self._action_dim}, "
            f"logit_base={self._logit_base}, gain={self._gain}, "
            f"W_seed={self._W_seed}, W_scale={self._W_scale})"
        )
