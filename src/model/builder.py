"""Model builders.

This module contains factory function responsible for
constructing neural network architectures."""

from __future__ import annotations

import timm
from torch import nn

from common.config.types import ModelConfig
from model.exceptions import ModelInitializationError


def build_timm_model(model_name: str, config: ModelConfig) -> nn.Module:
    """Build a TIMM classification model.

    Args:
        model_name:
            TIMM model architecture.

        config:
            Model configuration.

        Returns:
            Initialized neural network.

        Raises:
            ModelInitializationError:
                If model creation fails.
    """

    try:
        model = timm.create_model(
            model_name=model_name,
            pretrained=config.pretrained,
            num_classes=config.num_classes
        )

    except Exception as error:
        raise ModelInitializationError(
            f"Failed to initialize model "
            f"{model_name}"
        ) from error

    return model
        