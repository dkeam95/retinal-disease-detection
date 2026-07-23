"""Classification heads."""


from __future__ import annotations

from torch import nn

from common.config.types import ModelConfig


def build_classifier(in_features: int, config: ModelConfig) -> nn.Module:
    """
    Build classification head.

    Args:
        in_features:
            Number of backbone output features.

        config:
            Model configuration.

    Returns:
        Classification head.
    """

    return nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features, config.num_classes)
    )
