"""Observation wrappers for myoArm reaching state.

Layer position:

- Input: ``MyoArmState`` produced by an extractor or oracle path (true state).
- Output: ``MyoArmState`` representing what a *non-oracle* controller sees:
  delayed and / or noisy.
- These wrappers do NOT inherit from ``gymnasium.Wrapper``. They are pure
  ``MyoArmState -> MyoArmState`` transforms. Composition order is left to the
  caller; the env factory / experiment config decides whether to apply
  ``noisy(delayed(s))`` or ``delayed(noisy(s))``.

Internal representation:

- The public API takes and returns ``MyoArmState`` for type safety and
  readable field access.
- Internally each wrapper operates on the flat ``np.ndarray`` returned by
  ``MyoArmState.flatten()``, using ``StateSpec.layout()`` to address fields.
  This keeps the delay buffer compact and lets noise be applied per field
  without object copies.

Scope explicitly NOT covered by this module (handled later):

- raw Gym observation vector slicing
- ``gymnasium`` / ``myosuite`` import
- ``env.unwrapped`` extraction
- per-field clipping bounds for noisy observations
- ms-based delay specification (``dt``-aware ``from_ms`` factory)
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from myoarm_fse.envs.state import MyoArmState, StateSpec

_DTYPE: np.dtype = np.dtype(np.float32)
_VALID_FIELDS: tuple[str, ...] = (
    "qpos",
    "qvel",
    "act",
    "tip_pos",
    "target_pos",
    "reach_err",
)


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


def _check_state_matches_spec(state: MyoArmState, spec: StateSpec) -> None:
    if not isinstance(state, MyoArmState):
        raise ValueError(
            f"state must be MyoArmState, got {type(state).__name__}"
        )
    s = state.spec()
    if (s.qpos_dim, s.qvel_dim, s.act_dim) != (
        spec.qpos_dim,
        spec.qvel_dim,
        spec.act_dim,
    ):
        raise ValueError(
            f"state spec ({s.qpos_dim}, {s.qvel_dim}, {s.act_dim}) "
            f"does not match wrapper spec "
            f"({spec.qpos_dim}, {spec.qvel_dim}, {spec.act_dim})"
        )


# --- delay ---


class DelayedObservationWrapper:
    """Return the state seen ``delay_steps`` ago.

    On ``reset(initial_state)`` the internal ring buffer is filled with
    ``initial_state`` so that the first ``delay_steps`` observations are
    well-defined. Subsequent ``observe(s_t)`` calls push ``s_t`` and return
    the oldest entry.

    Timing
    ------
    With ``delay_steps = k``, after ``reset(s_0)``::

        observe(s_t) -> s_{t-k}     # treating s_0 as the reset state

    For example with ``k = 1``: ``observe(s_1) -> s_0``,
    ``observe(s_2) -> s_1``. With ``k = 2``: ``observe(s_1) -> s_0``,
    ``observe(s_2) -> s_0``, ``observe(s_3) -> s_1``.

    Internally the buffer is a ring of length ``k`` (not ``k + 1``); ``observe``
    returns the oldest entry and overwrites it with the new input.

    ``delay_steps == 0`` is the identity transform; ``reset`` is optional in
    that case. For ``delay_steps > 0`` calling ``observe`` before ``reset``
    raises ``RuntimeError``.
    """

    def __init__(self, spec: StateSpec, delay_steps: int) -> None:
        if not isinstance(spec, StateSpec):
            raise TypeError(
                f"spec must be a StateSpec, got {type(spec).__name__}"
            )
        _check_non_negative_int(delay_steps, "delay_steps")
        self._spec: StateSpec = spec
        self._delay_steps: int = delay_steps
        self._buffer: np.ndarray | None = None  # shape (delay_steps + 1, dim)
        self._head: int = 0  # index of the oldest entry in the ring

    @property
    def spec(self) -> StateSpec:
        return self._spec

    @property
    def delay_steps(self) -> int:
        return self._delay_steps

    def reset(self, initial_state: MyoArmState) -> None:
        _check_state_matches_spec(initial_state, self._spec)
        if self._delay_steps == 0:
            self._buffer = None
            return
        flat = initial_state.flatten()
        self._buffer = np.tile(flat, (self._delay_steps, 1)).astype(
            _DTYPE, copy=False
        )
        self._head = 0

    def observe(self, true_state: MyoArmState) -> MyoArmState:
        _check_state_matches_spec(true_state, self._spec)
        if self._delay_steps == 0:
            return true_state
        if self._buffer is None:
            raise RuntimeError(
                "DelayedObservationWrapper.observe called before reset; "
                "call reset(initial_state) at the start of each episode"
            )
        old = self._buffer[self._head].copy()
        self._buffer[self._head] = true_state.flatten()
        self._head = (self._head + 1) % self._buffer.shape[0]
        return MyoArmState.unflatten(old, self._spec)

    def __call__(self, true_state: MyoArmState) -> MyoArmState:
        return self.observe(true_state)

    def __repr__(self) -> str:
        return (
            f"DelayedObservationWrapper(spec={self._spec}, "
            f"delay_steps={self._delay_steps})"
        )


# --- noise ---


def _coerce_rng(rng: object) -> np.random.Generator:
    """Coerce ``rng`` to a ``numpy.random.Generator``. See ``noise._coerce_rng``."""
    if rng is None:
        return np.random.default_rng()
    if isinstance(rng, np.random.Generator):
        return rng
    if isinstance(rng, bool):
        raise TypeError(
            f"rng must be None, int, or np.random.Generator (got bool: {rng!r})"
        )
    if isinstance(rng, int):
        return np.random.default_rng(rng)
    raise TypeError(
        "rng must be None, int, or np.random.Generator, "
        f"got {type(rng).__name__}"
    )


def _validate_sigma_dict(sigma: Mapping[str, Any]) -> dict[str, float]:
    if not isinstance(sigma, Mapping):
        raise ValueError(
            f"sigma must be a mapping (dict), got {type(sigma).__name__}"
        )
    out: dict[str, float] = {}
    for key, value in sigma.items():
        if key not in _VALID_FIELDS:
            raise ValueError(
                f"unknown sigma field {key!r}; valid fields are {_VALID_FIELDS}"
            )
        if isinstance(value, bool):
            raise ValueError(
                f"sigma[{key!r}] must not be bool, got {value!r}"
            )
        if not isinstance(value, (int, float, np.floating)):
            raise ValueError(
                f"sigma[{key!r}] must be int, float, or np.floating, "
                f"got {type(value).__name__}"
            )
        v = float(value)
        if not np.isfinite(v):
            raise ValueError(f"sigma[{key!r}] must be finite, got {v}")
        if v < 0.0:
            raise ValueError(f"sigma[{key!r}] must be >= 0, got {v}")
        out[key] = v
    return out


class NoisyObservationWrapper:
    """Add per-field independent Gaussian noise to a ``MyoArmState``.

    Parameters
    ----------
    spec : StateSpec
        The state spec to validate inputs against.
    sigma : dict[str, float]
        Field name to noise scale. Unknown keys raise ``ValueError``. Missing
        keys are treated as ``0`` (no noise on that field).
    rng : None | int | np.random.Generator
        Random source. See :func:`_coerce_rng`.

    Notes
    -----
    Noise is additive Gaussian, applied independently per element::

        out_field = in_field + sigma_field * N(0, 1)

    No clipping is applied: ``qpos``, ``qvel``, ``tip_pos`` etc. have no
    natural unit-interval bound, and observation noise is conceptually
    distinct from command-side range constraints.

    Identity fast path: if ``sigma == {}`` or every entry is ``0``, the
    wrapper returns the input state's data unchanged and does not consume
    any randomness from ``rng``.
    """

    def __init__(
        self,
        spec: StateSpec,
        sigma: Mapping[str, float],
        rng: object = None,
    ) -> None:
        if not isinstance(spec, StateSpec):
            raise TypeError(
                f"spec must be a StateSpec, got {type(spec).__name__}"
            )
        self._spec: StateSpec = spec
        self._sigma: dict[str, float] = _validate_sigma_dict(sigma)
        self._rng: np.random.Generator = _coerce_rng(rng)
        self._is_identity: bool = (
            len(self._sigma) == 0 or all(v == 0.0 for v in self._sigma.values())
        )

    @property
    def spec(self) -> StateSpec:
        return self._spec

    @property
    def sigma(self) -> dict[str, float]:
        return dict(self._sigma)

    @property
    def rng(self) -> np.random.Generator:
        return self._rng

    def reset(self, seed: int | None = None) -> None:
        """Re-seed the internal RNG. ``seed=None`` draws a fresh independent rng."""
        if seed is None:
            self._rng = np.random.default_rng()
            return
        if isinstance(seed, bool):
            raise TypeError(f"seed must be None or int, got bool: {seed!r}")
        if not isinstance(seed, int):
            raise TypeError(
                f"seed must be None or int, got {type(seed).__name__}: {seed!r}"
            )
        self._rng = np.random.default_rng(seed)

    def observe(self, state: MyoArmState) -> MyoArmState:
        _check_state_matches_spec(state, self._spec)
        if self._is_identity:
            return state
        flat = state.flatten()
        layout = self._spec.layout()
        for name, sigma in self._sigma.items():
            if sigma == 0.0:
                continue
            slc = layout[name]
            n = slc.stop - slc.start
            noise = self._rng.standard_normal(size=n, dtype=np.float32) * np.float32(sigma)
            flat[slc] = flat[slc] + noise
        return MyoArmState.unflatten(flat, self._spec)

    def __call__(self, state: MyoArmState) -> MyoArmState:
        return self.observe(state)

    def __repr__(self) -> str:
        return (
            f"NoisyObservationWrapper(spec={self._spec}, sigma={self._sigma})"
        )
