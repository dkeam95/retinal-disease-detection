"""
Loss factory.
"""

from __future__ import annotations

from torch import Tensor
from torch import nn

from common.config.types import LossConfig

from losses.exceptions import (
    UnknownLossError,
)
from losses.loss_names import (
    LossName,
)
from losses.registry import (
    LOSS_REGISTRY,
)


def build_loss(
    config: LossConfig,
    class_weights: Tensor | None = None,
) -> nn.Module:
    """
    Build a loss function from configuration.

    Args:
        config:
            Loss configuration.

        class_weights:
            Optional class weights.

    Returns:
        Configured loss function.

    Raises:
        UnknownLossError:
            If the requested loss function is not supported.
    """

    try:
        loss_name = LossName(
            config.name,
        )

    except ValueError as error:
        available_losses = ", ".join(
            loss.value
            for loss in LossName
        )

        raise UnknownLossError(
            f"Unknown loss function: "
            f"{config.name!r}. "
            f"Available losses: "
            f"{available_losses}"
        ) from error

    builder = LOSS_REGISTRY[
        loss_name
    ]

    return builder(
        config,
        class_weights,
    )