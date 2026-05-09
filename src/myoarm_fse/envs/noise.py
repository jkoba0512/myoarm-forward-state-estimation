"""Signal-dependent motor noise applied to excitation.

Layer position (see ``actions.py`` for the full layer separation):

- Input: ``excitation_command`` ∈ ``[0, 1]^action_dim`` from the controller.
- Output: ``excitation`` ∈ ``[0, 1]^action_dim``, the canonical research input
  to muscle actuators after motor noise and clipping. ``excitation`` is what the
  forward model is trained on (see Q5 in the design notes).

The noise model (matching ``01_Project1`` Phase 0.3):

```text
noise = sigma * |u| * N(0, 1)        # element-wise independent
u_noisy = clip(u + noise, 0, 1)
```

Element-wise independent (no cross-muscle correlation). Variance scales with
``|u|`` so a silent muscle (``u = 0``) receives no noise.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

_DTYPE: np.dtype = np.dtype(np.float32)
_EXCITATION_LO: float = 0.0
_EXCITATION_HI: float = 1.0


def _coerce_rng(rng: object) -> np.random.Generator:
    """Coerce ``rng`` to a ``numpy.random.Generator``.

    Accepted forms:
    - ``None``: a fresh ``default_rng()`` (independent seed)
    - ``int``: ``default_rng(int)``
    - ``np.random.Generator``: returned as-is

    ``np.random.RandomState`` is intentionally not supported (legacy API).
    """
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


class SignalDependentMotorNoise:
    """Apply signal-dependent Gaussian motor noise to ``excitation_command``.

    Parameters
    ----------
    action_dim : int
        Number of muscles. Must be a positive ``int`` (``bool`` rejected).
    sigma : int | float | np.floating
        Noise scale. Must satisfy ``np.isfinite(sigma) and sigma >= 0``.
        ``bool`` is rejected. ``sigma == 0`` reduces ``apply`` to validation
        plus clipping; no noise is added.
    rng : None | int | np.random.Generator
        Random source. See :func:`_coerce_rng`.

    Notes
    -----
    Single-step 1-D only. Inputs are accepted as numpy array-likes
    (``np.ndarray``, ``list``, ``tuple``). Outputs are ``np.float32`` arrays
    of shape ``(action_dim,)``. Torch tensors are unsupported and untested;
    convert at the controller boundary.
    """

    def __init__(
        self,
        action_dim: int,
        sigma: float,
        rng: object = None,
    ) -> None:
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

        # sigma
        if isinstance(sigma, bool):
            raise ValueError(f"sigma must not be bool, got {sigma!r}")
        if not isinstance(sigma, (int, float, np.floating)):
            raise ValueError(
                "sigma must be int, float, or np.floating, "
                f"got {type(sigma).__name__}"
            )
        sigma_f = float(sigma)
        if not np.isfinite(sigma_f):
            raise ValueError(f"sigma must be finite, got {sigma_f}")
        if sigma_f < 0.0:
            raise ValueError(f"sigma must be >= 0, got {sigma_f}")

        self.action_dim: int = action_dim
        self._sigma: float = sigma_f
        self._rng: np.random.Generator = _coerce_rng(rng)

    @property
    def sigma(self) -> float:
        return self._sigma

    @property
    def rng(self) -> np.random.Generator:
        return self._rng

    def apply(self, excitation_command: npt.ArrayLike) -> np.ndarray:
        u = self._validate(excitation_command)
        if self._sigma > 0.0:
            gaussian = self._rng.standard_normal(size=u.shape, dtype=np.float32)
            noise = self._sigma * np.abs(u) * gaussian
            u = u + noise
        out = np.clip(u, _EXCITATION_LO, _EXCITATION_HI)
        return out.astype(_DTYPE, copy=False)

    def __call__(self, excitation_command: npt.ArrayLike) -> np.ndarray:
        return self.apply(excitation_command)

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

    def _validate(self, x: npt.ArrayLike) -> np.ndarray:
        arr = np.asarray(x, dtype=_DTYPE)
        if arr.ndim != 1:
            raise ValueError(
                f"excitation_command must be 1-D with shape ({self.action_dim},), "
                f"got ndim={arr.ndim} shape={arr.shape}"
            )
        if arr.shape[0] != self.action_dim:
            raise ValueError(
                f"excitation_command length {arr.shape[0]} does not match "
                f"action_dim={self.action_dim}"
            )
        if not np.isfinite(arr).all():
            raise ValueError(
                "excitation_command contains non-finite values (NaN or Inf)"
            )
        return arr

    def __repr__(self) -> str:
        return (
            f"SignalDependentMotorNoise(action_dim={self.action_dim}, "
            f"sigma={self._sigma})"
        )
