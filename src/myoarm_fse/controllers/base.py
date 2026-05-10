"""Controller Protocol.

Controllers consume a single ``MyoArmState`` (the observation visible to
the controller — possibly delayed/noisy) and emit an ``excitation_command``
in ``[0, 1]^action_dim``. This output is the input to the SDN layer in the
rollout pipeline (see ``data.rollout``); ActionAdapter and Gym/MuJoCo
mappings happen further downstream.

The protocol intentionally takes a ``MyoArmState`` (not a flat vector or
raw obs) so a controller can address fields by name. ``__call__`` is NOT
required — call sites use ``.act(...)`` only.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from myoarm_fse.envs.state import MyoArmState


@runtime_checkable
class Controller(Protocol):
    """Maps an observed ``MyoArmState`` to an excitation_command.

    Implementers must:

    - Expose ``action_dim`` (positive int).
    - Implement ``reset(seed=...)`` to (re)initialize internal state at the
      start of an episode. ``seed=None`` draws an independent rng.
    - Return a ``np.ndarray`` of shape ``(action_dim,)``, dtype
      ``np.float32``, with values in ``[0, 1]``.
    """

    @property
    def action_dim(self) -> int: ...

    def reset(self, *, seed: int | None = None) -> None: ...

    def act(self, observation: MyoArmState) -> np.ndarray: ...
