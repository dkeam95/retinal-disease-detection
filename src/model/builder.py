"""
Model builders.

This module contains factory functions responsible for
constructing neural network architectures.
"""

from __future__ import annotations

import timm
from torch import nn

from common.config.types import ModelConfig


def build_efficientnet_b0(config: ModelConfig) -> nn.Module:
    """
    Build an EfficientNet-B0 model.

    Parameters
    ----------
    config : ModelConfig
        Model configuration.

    Returns
    -------
    nn.Module
        Initialized EfficientNet-B0 model.
    """

    model = timm.create_model(
        model_name="efficientnet_b0",
        pretrained=config.pretrained,
        num_classes=config.num_classes,
    )

    return model