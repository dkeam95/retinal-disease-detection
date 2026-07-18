"""
Factory for constructing neural network models.

This module creates model instances based on a ModelConfig
using the registered model builders.
"""

from __future__ import annotations

import torch.nn as nn

from common.config.types import ModelConfig
from model.registry import MODEL_REGISTRY


def create_model(config: ModelConfig) -> nn.Module:
    """
    Create a neural network model from configuration.

    Parameters
    ----------
    config : ModelConfig
        Model configuration.

    Returns
    -------
    nn.Module
        Initialized neural network model.

    Raises
    ------
    ValueError
        If the requested model architecture is not registered.
    """

    try:
        builder = MODEL_REGISTRY[config.architecture]

    except KeyError as error:
        raise ValueError(
            f"Unsupported model architecture: "
            f"{config.architecture}"
        ) from error

    return builder(config)