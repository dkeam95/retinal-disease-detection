"""
Unit tests for the model backbone builder module.

This module contains unit tests verifying that `build_backbone` properly instantiates
feature extraction backbones based on the provided architecture name and configuration,
and that the returned backbone exposes the expected feature extraction interface.
"""

from __future__ import annotations  # Enables modern type hints (Python 3.7+)

from common.config.types import (
    ModelConfig,  # Dataclass defining model architecture configuration parameters
)
from model.backbone import (
    build_backbone,  # Function to instantiate feature extraction backbones
)


def test_build_backbone() -> None:
    """
    Verify that `build_backbone` creates a valid backbone instance with a callable feature extractor.
    """

    # Define model configuration without pretrained weights for fast testing
    config = ModelConfig(
        architecture="efficientnet_b0",
        pretrained=False,
        num_classes=5,
    )

    # Build backbone instance for efficientnet_b0
    backbone = build_backbone(
        "efficientnet_b0",
        config,
    )

    # Access forward_features method from created backbone
    features = backbone.forward_features

    # Assert forward_features is a callable method
    assert callable(
        features,
    )
