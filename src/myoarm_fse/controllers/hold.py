"""Static-hold excitation controller.

Outputs a constant excitation_command vector ``[value]*action_dim`` at
every step, regardless of the observation. ``value=0.0`` (default) means
silent muscles — the natural "no-control baseline" used to record the
forward dynamics under gravity alone or as a control case for forward
model dataset variety.
"""

from __future__ import annotations

import numpy as np

from myoarm_fse.envs.state import MyoArmState

_DTYPE: np.dtype = np.dtype(np.float32)
_LO: float = 0.0
_HI: float = 1.0


class HoldController:
    """Emit a constant excitation_command vector each step.

    Deterministic: ``act`` ignores the observation and ``reset`` is a
    no-op. ``reset`` still validates its ``seed`` argument so the
    Controller Protocol (whose ``reset(seed)`` accepts ``int | None``)
    surfaces type errors uniformly.
    """

    def __init__(self, action_dim: int, value: float = 0.0) -> None:
        # action_dim
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

        # value
        if isinstance(value, bool):
            raise ValueError(f"value must not be bool, got {value!r}")
        if not isinstance(value, (int, float, np.floating)):
            raise ValueError(
                f"value must be numeric, got {type(value).__name__}"
            )
        value_f = float(value)
        if not np.isfinite(value_f):
            raise ValueError(f"value must be finite, got {value_f}")
        if value_f < _LO or value_f > _HI:
            raise ValueError(
                f"value must be within [{_LO}, {_HI}], got {value_f}"
            )

        self._action_dim: int = action_dim
        self._value: float = value_f
        # Cache the output vector so .act() doesn't allocate per step.
        self._output: np.ndarray = np.full(action_dim, value_f, dtype=_DTYPE)

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def value(self) -> float:
        return self._value

    def reset(self, *, seed: int | None = None) -> None:
        if seed is None:
            return
        if isinstance(seed, bool):
            raise TypeError(f"seed must be None or int, got bool: {seed!r}")
        if not isinstance(seed, int):
            raise TypeError(
                f"seed must be None or int, got {type(seed).__name__}: {seed!r}"
            )
        # No internal randomness; the seed is accepted but ignored.

    def act(self, observation: MyoArmState) -> np.ndarray:
        if not isinstance(observation, MyoArmState):
            raise ValueError(
                f"observation must be MyoArmState, got {type(observation).__name__}"
            )
        # Return a fresh copy each call so callers cannot mutate the cache.
        return self._output.copy()

    def __repr__(self) -> str:
        return f"HoldController(action_dim={self._action_dim}, value={self._value})"
