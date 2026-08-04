"""
Backbone builders.

This module constructs feature extraction backbones.
"""

from __future__ import annotations  # Enables modern type hints (Python 3.7+)

import timm  # PyTorch Image Models library for pre-trained vision architectures
from torch import nn  # PyTorch neural network module base classes

from common.config.types import (
    ModelConfig,  # Strongly typed model configuration dataclass
)
from model.exceptions import (
    ModelInitializationError,  # Custom exception for model creation failures
)


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

    # Attempt to instantiate feature extractor via TIMM library
    try:
        backbone = timm.create_model(
            model_name=model_name,
            pretrained=config.pretrained,
            num_classes=0,  # Remove classification head to output raw feature embeddings
        )

    # Wrap any TIMM instantiation errors in domain-specific exception
    except Exception as error:
        raise ModelInitializationError(
            f"Failed to initialize backbone '{model_name}'."
        ) from error

    # Return constructed feature extractor module
    return backbone
