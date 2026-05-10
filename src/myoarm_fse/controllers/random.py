"""Random excitation controller.

Per-step independent clipped Gaussian noise around ``mean``. With
``mean=0.5, sigma=0.2`` (defaults), ~95% of samples land in ``[0.1, 0.9]``
before clipping; values outside ``[0, 1]`` are clipped silently. Used as
a baseline / smoke-test controller.

Setting ``sigma`` close to zero (e.g. 0.05) yields a "low-amplitude
random" controller per the Step 6 plan, without a separate class.
"""

from __future__ import annotations

import numpy as np

from myoarm_fse.envs.state import MyoArmState

_DTYPE: np.dtype = np.dtype(np.float32)
_LO: float = 0.0
_HI: float = 1.0


def _coerce_rng(rng: object) -> np.random.Generator:
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


class RandomController:
    """Per-step i.i.d. clipped-Gaussian excitation controller."""

    def __init__(
        self,
        action_dim: int,
        mean: float = 0.5,
        sigma: float = 0.2,
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

        # mean
        if isinstance(mean, bool):
            raise ValueError(f"mean must not be bool, got {mean!r}")
        if not isinstance(mean, (int, float, np.floating)):
            raise ValueError(
                f"mean must be numeric, got {type(mean).__name__}"
            )
        mean_f = float(mean)
        if not np.isfinite(mean_f):
            raise ValueError(f"mean must be finite, got {mean_f}")

        # sigma
        if isinstance(sigma, bool):
            raise ValueError(f"sigma must not be bool, got {sigma!r}")
        if not isinstance(sigma, (int, float, np.floating)):
            raise ValueError(
                f"sigma must be numeric, got {type(sigma).__name__}"
            )
        sigma_f = float(sigma)
        if not np.isfinite(sigma_f):
            raise ValueError(f"sigma must be finite, got {sigma_f}")
        if sigma_f < 0.0:
            raise ValueError(f"sigma must be >= 0, got {sigma_f}")

        self._action_dim: int = action_dim
        self._mean: float = mean_f
        self._sigma: float = sigma_f
        self._rng: np.random.Generator = _coerce_rng(rng)

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def mean(self) -> float:
        return self._mean

    @property
    def sigma(self) -> float:
        return self._sigma

    @property
    def rng(self) -> np.random.Generator:
        return self._rng

    def reset(self, *, seed: int | None = None) -> None:
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

    def act(self, observation: MyoArmState) -> np.ndarray:
        # observation is unused for this controller; the type check still
        # serves as a guard against accidental flat-array passes.
        if not isinstance(observation, MyoArmState):
            raise ValueError(
                f"observation must be MyoArmState, got {type(observation).__name__}"
            )
        x = self._rng.normal(self._mean, self._sigma, size=self._action_dim)
        return np.clip(x, _LO, _HI).astype(_DTYPE, copy=False)

    def __repr__(self) -> str:
        return (
            f"RandomController(action_dim={self._action_dim}, "
            f"mean={self._mean}, sigma={self._sigma})"
        )
