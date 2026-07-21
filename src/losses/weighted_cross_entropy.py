"""Weighted Cross Entropy loss builder."""

from __future__ import annotations          # Enables modern type hints (Python 3.7+)

from torch import Tensor                    # Type annotation for PyTorch multi-dimensional arrays
from torch import nn                        # PyTorch neural network modules and loss functions

from common.config.types import LossConfig  # Configuration object holding loss hyperparameters


def build_weighted_cross_entropy(config: LossConfig, class_weights: Tensor | None = None) -> nn.Module:
    """Build a weighted Cross Entropy loss.

    Args:
        config: Loss configuration.
        class_weights: Class weights.

    Returns:
        Weighted Cross Entropy loss instance.
    """
    if class_weights is None:
        raise ValueError("Class weights must be provided for weighted cross entropy.")
        
    # Instantiate PyTorch CrossEntropyLoss passing per-class weight Tensor and reduction strategy
    return nn.CrossEntropyLoss(
        weight=class_weights,
        reduction=config.reduction
    )