"""
Loss factory.
"""

from __future__ import annotations  # Enables modern type hints (Python 3.7+)

from torch import (
    Tensor,  # Type annotation for PyTorch Tensors
    nn,  # PyTorch base module class
)

from common.config.types import (
    LossConfig,  # Configuration dataclass for loss parameters
)
from losses.exceptions import (  # Exception raised when an invalid loss name is provided
    UnknownLossError,
)
from losses.loss_names import (  # Enum defining supported loss function identifiers
    LossName,
)
from losses.registry import (  # Dictionary mapping loss names to builder functions
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
        loss_name = LossName(  # Convert string name from config to LossName enum
            config.name,
        )

    except ValueError as error:
        available_losses = (
            ", ".join(  # Format list of available loss names for error message
                loss.value for loss in LossName
            )
        )

        raise UnknownLossError(  # Raise exception if loss name is not supported
            f"Unknown loss function: "
            f"{config.name!r}. "
            f"Available losses: "
            f"{available_losses}"
        ) from error

    builder = LOSS_REGISTRY[  # Retrieve loss builder function from registry
        loss_name
    ]

    return builder(  # Instantiate and return configured loss module
        config,
        class_weights,
    )
