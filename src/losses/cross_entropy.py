"""
Standard Cross Entropy loss builder.
"""

from __future__ import annotations  # Enables modern type hints (Python 3.7+)

from torch import (
    Tensor,  # Type annotation for PyTorch Tensors
    nn,  # PyTorch base module class
)

from common.config.types import (
    LossConfig,  # Configuration dataclass for loss parameters
)


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

    return nn.CrossEntropyLoss(                 # Instantiate standard PyTorch CrossEntropyLoss
        weight=class_weights,                   # Apply optional class weighting tensor
        reduction=config.reduction,             # Set reduction strategy (mean, sum, none)
    )
