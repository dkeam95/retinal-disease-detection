"""
Factory for constructing neural network models.

This module creates model instances based on a ModelConfig
using the registered model builders.
"""

from __future__ import annotations

import torch.nn as nn

from common.config.types import ModelConfig
from model.model_names import ModelArchitecture
from model.exceptions import UnknownModelArchitectureError
from model.registry import MODEL_REGISTRY


def create_model(
    config: ModelConfig,
) -> nn.Module:
    """
    Create a neural network model from configuration.

    Args:
        config:
            Model configuration.

    Returns:
        Initialized neural network model.

    Raises:
        ValueError:
            If the requested model architecture
            is not supported.
    """

    try:
        architecture = ModelArchitecture(
            config.architecture,
        )

    except ValueError as error:
        raise UnknownModelArchitectureError(
            f"Unsupported model architecture: "
            f"{config.architecture}"
        ) from error

    builder = MODEL_REGISTRY[
        architecture
    ]

    return builder(
        config,
    )