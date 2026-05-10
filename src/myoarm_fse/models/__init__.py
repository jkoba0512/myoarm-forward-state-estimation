"""Forward-model datasets and (later) the residual MLP / training loop."""

from myoarm_fse.models.datasets import (
    TransitionDataset,
    build_transitions,
    shuffle_transitions,
    split_by_episode,
)

__all__ = [
    "TransitionDataset",
    "build_transitions",
    "shuffle_transitions",
    "split_by_episode",
]
