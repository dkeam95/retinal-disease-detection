"""Loss factory."""

from __future__ import annotations

from torch import Tensor
from torch import nn

from common.config.types import LossConfig

from .registry import LOSS_REGISTRY


def build_loss(config: LossConfig, class_weights: Tensor | None = None) -> nn.Module:
    """Build a loss function from configuration.
    
    Args:
        config:
            Loss configuration.

        class_weights:
            Optional class weights.

    Returns:
        Configured loss function.

    Raises:
        ValueError:
            If the requested loss function is not registered.
    """

    builder = LOSS_REGISTRY.get(
        config.name
    )

    if builder is None:
        available_losses = ", ".join(
            sorted(LOSS_REGISTRY.keys())
        )

        raise ValueError(
            f"Unknown loss function: {config.name!r}. "
            f"Available losses: {available_losses}"
        )

    return builder(
        config,
        class_weights
    )