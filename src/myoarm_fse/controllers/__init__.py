"""Controllers that produce excitation_command from observations."""

from typing import Any

from myoarm_fse.controllers.base import Controller
from myoarm_fse.controllers.bc import (
    BCController,
    BCPolicy,
    BCTrainConfig,
    load_bc_policy,
    make_bc_model_id,
    save_bc_policy,
    state_feature_indices,
    train_bc_policy,
)
from myoarm_fse.controllers.heuristic_reach import HeuristicReachController
from myoarm_fse.controllers.hold import HoldController
from myoarm_fse.controllers.joint_pd import JointSpacePDController
from myoarm_fse.controllers.random import RandomController
from myoarm_fse.controllers.scripted_reach import ScriptedReachController

__all__ = [
    "BCController",
    "BCPolicy",
    "BCTrainConfig",
    "Controller",
    "HeuristicReachController",
    "HoldController",
    "JointSpacePDController",
    "RandomController",
    "ScriptedReachController",
    "load_bc_policy",
    "make_bc_model_id",
    "make_controller",
    "save_bc_policy",
    "state_feature_indices",
    "train_bc_policy",
]


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
    if name == "heuristic_reach":
        return HeuristicReachController(
            action_dim=action_dim,
            logit_base=float(spec.get("logit_base", 0.0)),
            gain=float(spec.get("gain", 5.0)),
            W_seed=int(spec.get("W_seed", seed)),
            W_scale=float(spec.get("W_scale", 1.0)),
        )
    raise ValueError(
        f"unknown controller name {name!r}; "
        "valid: 'random', 'hold', 'heuristic_reach'"
    )
