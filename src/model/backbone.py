"""
Backbone builders.

This module constructs feature extraction backbones.
"""

from __future__ import annotations

import timm
from torch import nn

from common.config.types import ModelConfig
from model.exceptions import ModelInitializationError


def build_backbone(model_name: str, config: ModelConfig) -> nn.Module:
    """
    Build a feature extraction backbone.

    Args:
        model_name:
            TIMM architecture name.

        config:
            Model configuration.

    Returns:
        Backbone model producing feature vectors.

    Raises:
        ModelInitializationError:
            If backbone initialization fails.
    """

    try:
        backbone = timm.create_model(
            model_name=model_name,
            pretrained=config.pretrained,
            num_classes=0,
        )

    except Exception as error:
        raise ModelInitializationError(
            f"Failed to initialize backbone "
            f"'{model_name}'."
        ) from error

    return backbone