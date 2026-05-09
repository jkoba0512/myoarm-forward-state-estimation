"""Pure state schema for myoArm reaching.

Layer position:

- This module is a *schema layer*. It validates, packs, and flattens named
  arrays into a single structured ``MyoArmState``.
- It does NOT read raw Gym observation vectors and does NOT import
  ``gymnasium`` / ``myosuite``. Extracting fields from a live env is a separate
  concern (``extract_state_from_env`` to be added later, alongside or near this
  module). Keeping the schema decoupled lets unit tests stay fast and stable
  while the env-side layout is still being verified.

Distinction between true state and controller-facing observation:

- ``MyoArmState`` is the *true* (oracle) state. The episode logger and metrics
  read it. Estimators consume noisy/delayed observations derived elsewhere.
- ``MyoArmState`` MUST NOT be passed to a controller in non-oracle baselines.
  Use a delayed/noisy observation (Step 4) for that path.

Field order is the single source of truth for ``flatten`` / ``unflatten``:

```text
qpos, qvel, act, tip_pos, target_pos, reach_err
```

``reach_err`` is the vector ``tip_pos - target_pos`` with shape ``(3,)``.
Scalar reach error is a metric, not a schema field; compute it via
``np.linalg.norm(state.reach_err)`` at the metrics layer.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

import numpy as np
import numpy.typing as npt

_DTYPE: np.dtype = np.dtype(np.float32)
_CART_DIM: int = 3

_FIELD_ORDER: tuple[str, ...] = (
    "qpos",
    "qvel",
    "act",
    "tip_pos",
    "target_pos",
    "reach_err",
)
_CART_FIELDS: frozenset[str] = frozenset({"tip_pos", "target_pos", "reach_err"})


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


@dataclass(frozen=True)
class StateSpec:
    """Dimension spec for a ``MyoArmState``.

    Cartesian fields (``tip_pos``, ``target_pos``, ``reach_err``) are fixed at
    shape ``(3,)`` and not part of the spec. Only the env-dependent dims are
    parameters here.
    """

    qpos_dim: int
    qvel_dim: int
    act_dim: int

    def __post_init__(self) -> None:
        _check_positive_int(self.qpos_dim, "qpos_dim")
        _check_positive_int(self.qvel_dim, "qvel_dim")
        _check_positive_int(self.act_dim, "act_dim")

    @property
    def dim(self) -> int:
        """Total flattened length: qpos + qvel + act + 3 * 3."""
        return self.qpos_dim + self.qvel_dim + self.act_dim + 3 * _CART_DIM

    def layout(self) -> dict[str, slice]:
        """Field-name → slice into a flat vector. Order matches ``_FIELD_ORDER``."""
        sizes = {
            "qpos": self.qpos_dim,
            "qvel": self.qvel_dim,
            "act": self.act_dim,
            "tip_pos": _CART_DIM,
            "target_pos": _CART_DIM,
            "reach_err": _CART_DIM,
        }
        out: dict[str, slice] = {}
        offset = 0
        for name in _FIELD_ORDER:
            n = sizes[name]
            out[name] = slice(offset, offset + n)
            offset += n
        return out


def _validate_1d(arr: np.ndarray, name: str) -> None:
    if arr.ndim != 1:
        raise ValueError(
            f"{name} must be 1-D, got ndim={arr.ndim} shape={arr.shape}"
        )


def _validate_cart(arr: np.ndarray, name: str) -> None:
    if arr.shape != (_CART_DIM,):
        raise ValueError(
            f"{name} must have shape (3,), got shape={arr.shape}"
        )


def _validate_finite(arr: np.ndarray, name: str) -> None:
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains non-finite values (NaN or Inf)")


def _coerce(x: npt.ArrayLike) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(x, dtype=_DTYPE))


@dataclass(frozen=True, eq=False)
class MyoArmState:
    """Structured myoArm reaching state (single step, true / oracle).

    All fields are ``np.ndarray`` of dtype ``np.float32``. ``qpos``, ``qvel``,
    and ``act`` are 1-D with env-dependent length; ``tip_pos``, ``target_pos``,
    and ``reach_err`` are shape ``(3,)``.

    Construct via :meth:`from_arrays` (which coerces array-likes to float32)
    or directly with ``np.ndarray`` inputs. ``__post_init__`` enforces dtype,
    shape, and finiteness; out-of-spec inputs raise ``ValueError``. Field
    declaration order is the flatten order.
    """

    qpos: np.ndarray
    qvel: np.ndarray
    act: np.ndarray
    tip_pos: np.ndarray
    target_pos: np.ndarray
    reach_err: np.ndarray

    def __post_init__(self) -> None:
        for f in fields(self):
            value = getattr(self, f.name)
            if not isinstance(value, np.ndarray):
                raise ValueError(
                    f"{f.name} must be np.ndarray (use MyoArmState.from_arrays "
                    f"to coerce array-likes), got {type(value).__name__}"
                )
            if value.dtype != _DTYPE:
                raise ValueError(
                    f"{f.name} must have dtype float32, got {value.dtype}"
                )
            if f.name in _CART_FIELDS:
                _validate_cart(value, f.name)
            else:
                _validate_1d(value, f.name)
            _validate_finite(value, f.name)

    @classmethod
    def from_arrays(
        cls,
        qpos: npt.ArrayLike,
        qvel: npt.ArrayLike,
        act: npt.ArrayLike,
        tip_pos: npt.ArrayLike,
        target_pos: npt.ArrayLike,
        reach_err: npt.ArrayLike,
    ) -> MyoArmState:
        """Coerce array-likes to ``np.float32`` arrays and validate."""
        return cls(
            qpos=_coerce(qpos),
            qvel=_coerce(qvel),
            act=_coerce(act),
            tip_pos=_coerce(tip_pos),
            target_pos=_coerce(target_pos),
            reach_err=_coerce(reach_err),
        )

    def spec(self) -> StateSpec:
        return StateSpec(
            qpos_dim=int(self.qpos.shape[0]),
            qvel_dim=int(self.qvel.shape[0]),
            act_dim=int(self.act.shape[0]),
        )

    def flatten(self) -> np.ndarray:
        """Concatenate all fields in declaration order. dtype float32."""
        return np.concatenate(
            [getattr(self, name) for name in _FIELD_ORDER],
            dtype=_DTYPE,
        )

    @classmethod
    def unflatten(cls, vec: npt.ArrayLike, spec: StateSpec) -> MyoArmState:
        """Inverse of :meth:`flatten` given a ``StateSpec``."""
        if not isinstance(spec, StateSpec):
            raise TypeError(
                f"spec must be a StateSpec, got {type(spec).__name__}"
            )
        arr = np.asarray(vec, dtype=_DTYPE)
        if arr.ndim != 1:
            raise ValueError(
                f"vec must be 1-D, got ndim={arr.ndim} shape={arr.shape}"
            )
        if arr.shape[0] != spec.dim:
            raise ValueError(
                f"vec length {arr.shape[0]} does not match spec.dim={spec.dim}"
            )
        layout = spec.layout()
        return cls.from_arrays(
            qpos=arr[layout["qpos"]],
            qvel=arr[layout["qvel"]],
            act=arr[layout["act"]],
            tip_pos=arr[layout["tip_pos"]],
            target_pos=arr[layout["target_pos"]],
            reach_err=arr[layout["reach_err"]],
        )
