"""Tests for the model factory."""

from __future__ import annotations  # Enables modern type hints (Python 3.7+)

import pytest  # Testing framework for asserting exceptions and test execution
from torch import nn  # PyTorch base module class

from common.config.types import ModelConfig  # Dataclass defining model architecture parameters
from model.factory import create_model  # Factory function that instantiates PyTorch models


def test_create_efficientnet_b0() -> None:
    """Factory should create an EfficientNet-B0 model."""

    # Configuration specifying EfficientNet-B0 architecture with custom class count
    config = ModelConfig(
        architecture="efficientnet_b0",
        pretrained=False,  # Disable loading pre-trained weights for fast unit testing
        num_classes=5
    )

    # Instantiate model using the factory function
    model = create_model(config)

    # Verify that the returned object inherits from torch.nn.Module
    assert isinstance(model, nn.Module)


def test_unknown_architecture() -> None:
    """Factory should raise ValueError for unknown architecture."""

    # Configuration with an unsupported architecture identifier
    config = ModelConfig(
        architecture="unknown",
        pretrained=False,
        num_classes=5
    )

    # Verify that passing an invalid architecture raises a ValueError
    with pytest.raises(ValueError):
        create_model(config)