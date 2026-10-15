"""Training: one unified loop plus the single / cross-condition protocols."""

from .loop import Trainer, TrainResult, load_checkpoint
from .protocols import (
    build_for_artifact,
    checkpoint_path,
    model_kwargs_from_config,
    should_use_mmd,
    target_split_for_mmd,
    train_one,
)

__all__ = [
    "Trainer",
    "TrainResult",
    "load_checkpoint",
    "train_one",
    "build_for_artifact",
    "model_kwargs_from_config",
    "target_split_for_mmd",
    "checkpoint_path",
    "should_use_mmd",
]
