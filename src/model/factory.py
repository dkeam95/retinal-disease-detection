"""
Factory for constructing neural network models.

This module creates model instances based on a ModelConfig
using the registered model builders.
"""

from __future__ import annotations           # Enables modern type hints (Python 3.7+)

import torch.nn as nn                        # Neural network modules base class

from common.config.types import ModelConfig  # Configuration object holding architecture parameters
from model.registry import MODEL_REGISTRY    # Global registry dictionary mapping architecture names to builder functions


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
        # Retrieve the registered builder function using the architecture key from config
        builder = MODEL_REGISTRY[config.architecture]

    except KeyError as error:
        # Raise an informative ValueError if the architecture string is not present in the registry
        raise ValueError(
            f"Unsupported model architecture: "
            f"{config.architecture}"
        ) from error

    # Execute the selected builder function, passing the configuration object to instantiate the model
    return builder(config)