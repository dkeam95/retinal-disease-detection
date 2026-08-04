"""
Weighted Cross Entropy loss builder.
"""

from __future__ import annotations  # Enables modern type hints (Python 3.7+)

from torch import (
    Tensor,  # Type annotation for PyTorch Tensors
    nn,  # PyTorch base module class
)

from common.config.types import (
    LossConfig,  # Configuration dataclass for loss parameters
)
from losses.exceptions import (  # Exception raised when loss setup fails
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
        raise LossInitializationError(  # Ensure class weights tensor is explicitly provided
            "Weighted Cross Entropy requires class weights."
        )

    return nn.CrossEntropyLoss(  # Instantiate weighted PyTorch CrossEntropyLoss
        weight=class_weights,  # Pass required class weighting tensor
        reduction=config.reduction,  # Set reduction strategy (mean, sum, none)
    )
