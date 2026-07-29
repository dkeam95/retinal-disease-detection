"""
Tests for the model factory module.

This module contains unit tests verifying that model creation functions correctly
instantiate supported model architectures and handle invalid architecture names appropriately.
"""

from __future__ import annotations  # Enables modern type hints (Python 3.7+)

import pytest
from torch import nn

from common.config.types import ModelConfig
from model.exceptions import UnknownModelArchitectureError
from model.factory import create_model


def test_create_efficientnet_b0() -> None:
    """
    Verify that the factory correctly builds an EfficientNet-B0 PyTorch module.
    """

    # Initialize model configuration for non-pretrained EfficientNet-B0 with 5 classes
    config = ModelConfig(
        architecture="efficientnet_b0",
        pretrained=False,
        num_classes=5,
    )

    # Instantiate the neural network model via the factory
    model = create_model(config)

    # Assert that the created model is a valid PyTorch nn.Module instance
    assert isinstance(model, nn.Module)


def test_unknown_architecture() -> None:
    """
    Verify that requesting an unsupported architecture raises UnknownModelArchitectureError.
    """

    # Initialize configuration with an unregistered model architecture name
    config = ModelConfig(
        architecture="unknown",
        pretrained=False,
        num_classes=5,
    )

    # Assert that UnknownModelArchitectureError is raised with the expected error message
    with pytest.raises(
        UnknownModelArchitectureError,
        match="Unsupported model architecture",
    ):
        create_model(config)
