"""
Factory for constructing neural network models.

This module creates model instances based on a ModelConfig
using the registered model builders.
"""

from __future__ import annotations  # Enables modern type hints (Python 3.7+)

import torch.nn as nn  # PyTorch neural network module base classes

from common.config.types import (
    ModelConfig,  # Strongly typed model configuration dataclass
)
from model.exceptions import (
    UnknownModelArchitectureError,  # Domain-specific exception for invalid architectures
)
from model.model_names import (
    ModelArchitecture,  # Enum defining supported model architectures
)
from model.registry import (
    MODEL_REGISTRY,  # Central dispatch table mapping architectures to builder functions
)


def create_model(
    config: ModelConfig,
) -> nn.Module:
    """Create a neural network model from configuration.

    Args:
        config:
            Model configuration.

    Returns:
        Initialized neural network model.

    Raises:
        UnknownModelArchitectureError:
            If the requested model architecture is not supported.
    """

    # Validate string architecture name by mapping to ModelArchitecture enum
    try:
        architecture = ModelArchitecture(
            config.architecture,
        )

    # Wrap enum parsing failure in domain-specific exception
    except ValueError as error:
        raise UnknownModelArchitectureError(
            f"Unsupported model architecture: "
            f"{config.architecture}"
        ) from error

    # Retrieve registered builder function for requested architecture
    builder = MODEL_REGISTRY[
        architecture
    ]

    # Instantiate and return neural network model
    return builder(
        config,
    )
