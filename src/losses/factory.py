"""Loss factory."""

from __future__ import annotations  # Enables modern type hints (Python 3.7+)

from torch import Tensor            # PyTorch Tensor type hint
from torch import nn                # Base class for neural network loss modules

from common.config.types import LossConfig  # Configuration object holding loss hyperparameters

from .registry import LOSS_REGISTRY  # Global registry mapping loss names to builder functions


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

    # Retrieve the builder function from registry using loss name key from configuration
    builder = LOSS_REGISTRY.get(
        config.name
    )

    # Handle unregistered loss types by raising a informative ValueError with available choices
    if builder is None:
        available_losses = ", ".join(
            sorted(LOSS_REGISTRY.keys())
        )

        raise ValueError(
            f"Unknown loss function: {config.name!r}. "
            f"Available losses: {available_losses}"
        )

    # Execute selected loss builder passing config parameters and optional class weights
    return builder(
        config,
        class_weights
    )