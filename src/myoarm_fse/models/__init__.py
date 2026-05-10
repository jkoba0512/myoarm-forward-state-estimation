"""Forward-model datasets, residual MLP, and training loop."""

from myoarm_fse.models.datasets import (
    TransitionDataset,
    build_transitions,
    concat_datasets,
    shuffle_transitions,
    split_by_episode,
    split_by_local_indices,
)
from myoarm_fse.models.mlp import ForwardMLP
from myoarm_fse.models.train import (
    TrainConfig,
    load_model,
    make_model_id,
    make_train_val_split,
    rollout_predictions,
    save_model,
    setup_seeds,
    train_forward_model,
)

__all__ = [
    # datasets
    "TransitionDataset",
    "build_transitions",
    "concat_datasets",
    "shuffle_transitions",
    "split_by_episode",
    "split_by_local_indices",
    # mlp
    "ForwardMLP",
    # train
    "TrainConfig",
    "load_model",
    "make_model_id",
    "make_train_val_split",
    "rollout_predictions",
    "save_model",
    "setup_seeds",
    "train_forward_model",
]
