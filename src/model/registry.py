"""
Registry of available model builders.

This module maps model architecture names to their
corresponding builder functions.
"""

from __future__ import annotations     # Enables modern type hints (Python 3.7+)

from collections.abc import Callable   # Type hint for callable builder functions

from torch import nn                   # Neural network modules base class

from common.config.types import ModelConfig      # Configuration object holding architecture parameters
from model.builder import build_efficientnet_b0  # Builder function for EfficientNet-B0 architecture


# Type alias for builder functions accepting ModelConfig and returning a PyTorch Module
ModelBuilder = Callable[
    [ModelConfig],
    nn.Module,
]

# Registry mapping architecture key strings to their factory builder functions
MODEL_REGISTRY: dict[str, ModelBuilder] = {
    "efficientnet_b0": build_efficientnet_b0,
}