"""Controllers that produce excitation_command from observations."""

from typing import Any

from myoarm_fse.controllers.base import Controller
from myoarm_fse.controllers.hold import HoldController
from myoarm_fse.controllers.random import RandomController

__all__ = ["Controller", "HoldController", "RandomController", "make_controller"]


def make_controller(
    spec: dict[str, Any],
    action_dim: int,
    seed: int,
) -> Controller:
    """Build a Controller from a config sub-dict.

    ``spec["name"]`` selects the class; remaining keys are passed as
    kwargs. ``seed`` is consumed by random controllers; deterministic
    controllers (e.g. ``HoldController``) accept the argument for
    interface uniformity but ignore it.
    """
    if not isinstance(spec, dict):
        raise ValueError(f"controller spec must be dict, got {type(spec).__name__}")
    if "name" not in spec:
        raise ValueError(f"controller spec missing 'name' key, got {spec!r}")
    name = spec["name"]
    if name == "random":
        return RandomController(
            action_dim=action_dim,
            mean=float(spec.get("mean", 0.5)),
            sigma=float(spec.get("sigma", 0.2)),
            rng=seed,
        )
    if name == "hold":
        return HoldController(
            action_dim=action_dim,
            value=float(spec.get("value", 0.0)),
        )
    raise ValueError(
        f"unknown controller name {name!r}; valid: 'random', 'hold'"
    )
