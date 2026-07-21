"""
Model builders.

This module contains factory functions responsible for
constructing neural network architectures.
"""

from __future__ import annotations           # Enables modern type hints (Python 3.7+)

import timm                                  # PyTorch Image Models library providing pre-trained computer vision architectures
from torch import nn                         # Neural network modules base class

from common.config.types import ModelConfig  # Configuration object holding architecture hyperparameters


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

    # Instantiate EfficientNet-B0 model via timm with specified weights and classification head dimensions
    model = timm.create_model(
        model_name="efficientnet_b0",
        pretrained=config.pretrained,    # Download/load pre-trained ImageNet weights if True
        num_classes=config.num_classes,  # Adjust the output dimension of the final fully-connected layer
    )

    return model