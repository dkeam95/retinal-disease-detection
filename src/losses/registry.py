"""Loss function registry."""

from __future__ import annotations  # Enables modern type hints (Python 3.7+)

from collections.abc import Callable  # Abstract base class for type hinting callable objects

from torch import Tensor  # PyTorch Tensor type annotation
from torch import nn  # Neural network base module

from common.config.types import LossConfig  # Loss configuration dataclass

# Import factory builder functions for all supported loss implementations
from .class_balanced_focal import (
    build_class_balanced_focal_loss
)
from .cross_entropy import (
    build_cross_entropy
)
from .focal import (
    build_focal_loss
)
from .weighted_cross_entropy import (
    build_weighted_cross_entropy
)


# Define type alias for loss builder signatures: accepts LossConfig and optional class_weights Tensor, returns PyTorch nn.Module
LossBuilder = Callable[
    [LossConfig, Tensor | None],
    nn.Module
]


# Global dictionary registry mapping string identifiers to their respective builder functions
LOSS_REGISTRY: dict[str, LossBuilder] = {
    "cross_entropy": build_cross_entropy,
    "weighted_cross_entropy": build_weighted_cross_entropy,
    "focal": build_focal_loss,
    "class_balanced_focal": build_class_balanced_focal_loss,
}