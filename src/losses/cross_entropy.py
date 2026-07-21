"""Standart Cross Entropy loss builder."""

from __future__ import annotations          # Enables modern type hints (Python 3.7+)

from torch import nn                        # PyTorch neural network modules and loss functions

from torch import Tensor                    # PyTorch tensor class
from common.config.types import LossConfig  # Configuration object holding loss hyperparameters


def build_cross_entropy(config: LossConfig, class_weights: Tensor | None = None) -> nn.Module:
    """Build the standart Cross Entropy loss.

    Args:
        config: Loss configuration.

    Returns:
        Cross Entropy loss instance.
    """
    # Instantiate standard PyTorch CrossEntropyLoss passing the reduction mode ('mean', 'sum', or 'none')
    return nn.CrossEntropyLoss(
        reduction=config.reduction
    )