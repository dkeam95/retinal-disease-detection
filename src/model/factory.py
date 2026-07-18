"""Model factory ."""

from __future__ import annotations

import torch.nn as nn
import timm

from common.config.types import ModelConfig
from model.registry import MODEL_REGISTRY


def create_model(config: ModelConfig) -> nn.Module:
    """Create a model from configuration.

    Args:
        config: Model configuration.

    Returns:
        Initialized model.

    Raises:
        ValueError: if the requested architecture is not registered.
    """

    try:
        builder = MODEL_REGISTRY[config.architecture]
    
    except KeyError as error:
        raise ValueError(
            f"Unsupported model architecture: {config.architecture}"
        ) from error

    return builder(config)

    
    
