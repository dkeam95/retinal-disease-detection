"""
Unit tests for the model builder module using the `timm` library.

This module contains unit tests verifying that `build_timm_model` correctly instantiates
neural network architectures according to the specified configuration parameters.
"""

from __future__ import annotations  # Enables modern type hints (Python 3.7+)

from common.config.types import (
    ModelConfig,  # Dataclass defining model architecture configuration parameters
)
from model.builder import (
    build_timm_model,  # Function to instantiate model using timm library
)


def test_build_timm_model() -> None:
    """
    Verify that `build_timm_model` successfully creates a valid model instance.
    """

    # Define model configuration for efficientnet_b0 without pretrained weights for fast testing
    config = ModelConfig(
        architecture="efficientnet_b0",
        pretrained=False,
        num_classes=5,
    )

    # Instantiate model instance via timm builder
    model = build_timm_model(
        "efficientnet_b0",
        config,
    )

    # Verify model object was successfully instantiated
    assert model is not None

    # Print class name of created model architecture
    print(model.__class__.__name__)
