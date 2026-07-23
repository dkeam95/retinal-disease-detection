"""
Standard Cross Entropy loss builder.
"""

from __future__ import annotations

from torch import Tensor
from torch import nn

from common.config.types import LossConfig


def build_cross_entropy(
    config: LossConfig,
    class_weights: Tensor | None = None,
) -> nn.Module:
    """
    Build the standard Cross Entropy loss.

    Args:
        config:
            Loss configuration.

        class_weights:
            Optional class weights.

    Returns:
        Configured Cross Entropy loss.
    """

    return nn.CrossEntropyLoss(
        weight=class_weights,
        reduction=config.reduction,
    )