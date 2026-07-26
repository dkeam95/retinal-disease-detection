"""
Unit tests for model data types and output containers.

This module contains unit tests verifying that `ModelOutput` containers correctly store
and expose intermediate tensors (logits and features) produced during model inference.
"""

from __future__ import annotations  # Enables modern type hints (Python 3.7+)

import torch  # PyTorch tensor library

from model.types import ModelOutput  # Data structure/dataclass container for model outputs


def test_model_output() -> None:
    """
    Verify that `ModelOutput` correctly initializes and retains logits and feature tensors.
    """

    # Generate dummy logits tensor (batch_size=4, num_classes=5)
    logits = torch.randn(
        4,
        5,
    )

    # Generate dummy extracted features tensor (batch_size=4, feature_dim=1280)
    features = torch.randn(
        4,
        1280,
    )

    # Instantiate ModelOutput container with logits and features
    output = ModelOutput(
        logits=logits,
        features=features,
    )

    # Verify logits attribute holds correct tensor shape
    assert output.logits.shape == (
        4,
        5,
    )

    # Verify features attribute holds correct tensor shape
    assert output.features.shape == (
        4,
        1280,
    )

    # Print string representation of the ModelOutput instance
    print(output)