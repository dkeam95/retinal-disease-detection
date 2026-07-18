"""
Registry of available model builders.

This module maps model architecture names to their
corresponding builder functions.
"""

from __future__ import annotations

from collections.abc import Callable

from torch import nn

from common.config.types import ModelConfig
from model.builder import build_efficientnet_b0


ModelBuilder = Callable[
    [ModelConfig],
    nn.Module,
]

# Registry of supported model architectures.
MODEL_REGISTRY: dict[str, ModelBuilder] = {
    "efficientnet_b0": build_efficientnet_b0,
}