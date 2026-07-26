"""Classification heads.

This module is responsible for building classification heads.
"""

from __future__ import annotations  # Enables modern type hints (Python 3.7+)

from torch import nn  # PyTorch neural network module base classes

from common.config.types import ModelConfig  # Strongly typed model configuration dataclass


def build_classifier(in_features: int, config: ModelConfig) -> nn.Module:
    """Build classification head.

    Args:
        in_features:
            Number of backbone output features.

        config:
            Model configuration.

    Returns:
        Classification head module containing Dropout and Linear projection layers.
    """

    # Assemble simple linear classification head with dropout regularization
    return nn.Sequential(
        nn.Dropout(p=0.2),  # Dropout layer with 20% probability for regularization
        nn.Linear(in_features, config.num_classes),  # Linear projection from backbone embeddings to target classes
    )