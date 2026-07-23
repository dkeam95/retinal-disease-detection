"""
Weighted Cross Entropy loss builder.
"""

from __future__ import annotations

from torch import Tensor
from torch import nn

from common.config.types import LossConfig

from losses.exceptions import (
    LossInitializationError,
)


def build_weighted_cross_entropy(
    config: LossConfig,
    class_weights: Tensor | None = None,
) -> nn.Module:
    """
    Build Weighted Cross Entropy loss.

    Args:
        config:
            Loss configuration.

        class_weights:
            Class weights.

    Returns:
        Configured Weighted Cross Entropy loss.

    Raises:
        LossInitializationError:
            If class weights are not provided.
    """

    if class_weights is None:
        raise LossInitializationError(
            "Weighted Cross Entropy requires "
            "class weights."
        )

    return nn.CrossEntropyLoss(
        weight=class_weights,
        reduction=config.reduction,
    )