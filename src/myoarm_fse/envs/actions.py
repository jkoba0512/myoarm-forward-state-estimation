"""Action adapter between research-canonical excitation and Gym/MyoSuite api_action.

Layer separation kept by this module (do not collapse them elsewhere):

- ``neural_command``: abstract motor command from a controller. Translation to
  ``excitation`` is the controller's responsibility, not this adapter's.
- ``excitation``: canonical research input to muscle actuators in
  ``[0, 1]^action_dim``. Shared representation across controllers, datasets,
  and forward models.
- ``api_action``: input to ``gym.Env.step()`` / MyoSuite, expected in
  ``[-1, 1]^action_dim``.
- ``activation``: muscle-internal activation state inside MuJoCo. NOT produced
  or consumed by this adapter.
- ``ctrl``: final actuator control written to ``mj_data.ctrl`` by the env after
  ``step()``. Inspected post-step from the simulator by the episode logger.

This adapter ONLY converts between ``excitation`` and ``api_action`` and clips
each to its valid range. Activation and ctrl are read after ``env.step()``.

Inputs are accepted as numpy array-likes (``np.ndarray``, ``list``, ``tuple``).
Torch tensors are unsupported and untested; convert to numpy at the controller
boundary (``tensor.detach().cpu().numpy()``).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

_DTYPE: np.dtype = np.dtype(np.float32)

_EXCITATION_LO: float = 0.0
_EXCITATION_HI: float = 1.0
_API_ACTION_LO: float = -1.0
_API_ACTION_HI: float = 1.0


class ActionAdapter:
    """Convert between excitation in ``[0, 1]^n`` and api_action in ``[-1, 1]^n``.

    Single-step 1-D only. Outputs are ``np.float32`` arrays of shape
    ``(action_dim,)``. Out-of-range values are clipped silently; non-finite
    values raise ``ValueError``.
    """

    def __init__(self, action_dim: int) -> None:
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
        self.action_dim: int = action_dim

    def excitation_to_api_action(self, excitation: npt.ArrayLike) -> np.ndarray:
        u = self._validate_and_clip(
            excitation, name="excitation", lo=_EXCITATION_LO, hi=_EXCITATION_HI
        )
        return (2.0 * u - 1.0).astype(_DTYPE, copy=False)

    def api_action_to_excitation(self, api_action: npt.ArrayLike) -> np.ndarray:
        a = self._validate_and_clip(
            api_action, name="api_action", lo=_API_ACTION_LO, hi=_API_ACTION_HI
        )
        return (0.5 * (a + 1.0)).astype(_DTYPE, copy=False)

    def clip_excitation(self, excitation: npt.ArrayLike) -> np.ndarray:
        return self._validate_and_clip(
            excitation, name="excitation", lo=_EXCITATION_LO, hi=_EXCITATION_HI
        )

    def clip_api_action(self, api_action: npt.ArrayLike) -> np.ndarray:
        return self._validate_and_clip(
            api_action, name="api_action", lo=_API_ACTION_LO, hi=_API_ACTION_HI
        )

    def _validate_and_clip(
        self, x: npt.ArrayLike, name: str, lo: float, hi: float
    ) -> np.ndarray:
        arr = np.asarray(x, dtype=_DTYPE)
        if arr.ndim != 1:
            raise ValueError(
                f"{name} must be 1-D with shape ({self.action_dim},), "
                f"got ndim={arr.ndim} shape={arr.shape}"
            )
        if arr.shape[0] != self.action_dim:
            raise ValueError(
                f"{name} length {arr.shape[0]} does not match "
                f"action_dim={self.action_dim}"
            )
        if not np.isfinite(arr).all():
            raise ValueError(f"{name} contains non-finite values (NaN or Inf)")
        return np.clip(arr, lo, hi).astype(_DTYPE, copy=False)

    def __repr__(self) -> str:
        return f"ActionAdapter(action_dim={self.action_dim})"


def detect_action_dim(env: Any) -> int:
    """Return ``action_dim`` from a Gym-like env via ``env.action_space.shape``.

    Accepts any object exposing ``.action_space`` with a 1-D ``.shape``. Does
    not import ``gymnasium`` so callers (and tests) can pass minimal stubs.

    Range (``low``/``high``) is intentionally NOT validated here; pinning
    ``normalize_act=True`` is the env factory's responsibility.
    """
    try:
        space = env.action_space
    except AttributeError as exc:
        raise TypeError("env has no .action_space attribute") from exc
    shape = getattr(space, "shape", None)
    if shape is None:
        raise ValueError("env.action_space has no .shape attribute")
    if len(shape) != 1:
        raise ValueError(
            f"expected 1-D action_space.shape, got shape={shape!r}; "
            "myoArm uses a flat muscle excitation vector"
        )
    dim = int(shape[0])
    if dim <= 0:
        raise ValueError(f"action_space.shape[0] must be > 0, got {dim}")
    return dim
