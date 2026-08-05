"""Classification heads.

This module is responsible for building classification heads.
"""

from __future__ import annotations  # Enables modern type hints (Python 3.7+)

from torch import nn  # PyTorch neural network module base classes

from common.config.types import (
    ModelConfig,  # Strongly typed model configuration dataclass
)


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
    dropout_rate = getattr(config, "dropout_rate", 0.0)
    linear = nn.Linear(in_features, config.num_classes)

    # Weight/bias initialization (Xavier Uniform / Zeros)
    nn.init.xavier_uniform_(linear.weight)
    if linear.bias is not None:
        nn.init.zeros_(linear.bias)

    # Assemble simple linear classification head with dropout regularization
    return nn.Sequential(
        nn.Dropout(p=dropout_rate),
        linear,
    )
