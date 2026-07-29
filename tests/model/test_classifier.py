"""
Unit tests for the classifier head builder module.

This module contains unit tests verifying that `build_classifier` properly instantiates
classification heads with correct input/output feature dimensions matching the provided
model configuration.
"""

from __future__ import annotations  # Enables modern type hints (Python 3.7+)

import torch  # PyTorch tensor library

from common.config.types import (
    ModelConfig,  # Dataclass defining model architecture configuration parameters
)
from model.classifier import (
    build_classifier,  # Utility function to construct classifier head
)


def test_build_classifier() -> None:
    """
    Verify that `build_classifier` instantiates a functional classifier head producing correct output shapes.
    """

    # Define model configuration specifying target class count
    config = ModelConfig(
        architecture="efficientnet_b0",
        pretrained=False,  # Disable pretrained weights loading for fast testing
        num_classes=5,  # Set target output dimensions to 5 classes
    )

    # Instantiate classifier head for feature dimension of 1280
    classifier = build_classifier(
        in_features=1280,
        config=config,
    )

    # Generate dummy input tensor representing feature batch (batch_size=4, in_features=1280)
    x = torch.randn(
        4,
        1280,
    )

    # Perform forward pass through classifier head
    logits = classifier(
        x,
    )

    # Assert output shape matches expected (batch_size, num_classes)
    assert logits.shape == (
        4,
        5,
    )

    # Print output shape of resulting logits tensor
    print(logits.shape)
