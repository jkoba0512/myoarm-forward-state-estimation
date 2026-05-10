"""Estimator Protocol.

State estimators consume a sequence of ``(y_obs, u)`` and produce a
running point estimate of the underlying state. The protocol is
deliberately point-estimate only — covariance / particles are added by
specific estimator implementations (Phase 3.2 learned gain, Project C
particle filter) without changing this base contract.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Estimator(Protocol):
    """Sequential state estimator over flat (state_dim,) observations.

    Implementers must:

    - Expose ``state_dim`` (positive int).
    - Implement ``reset(initial_state)`` to seed the internal estimate
      at the start of an episode.
    - Return a fresh ``np.ndarray`` of shape ``(state_dim,)`` and
      dtype ``np.float32`` from ``step``.
    """

    @property
    def state_dim(self) -> int: ...

    def reset(self, initial_state: np.ndarray) -> None: ...

    def step(self, y_obs: np.ndarray, u: np.ndarray) -> np.ndarray: ...
