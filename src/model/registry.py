"""Registry of available model builders.

This module maps model architecture names to their
corresponding builder functions.
"""

from __future__ import annotations  # Enables modern type hints (Python 3.7+)

from collections.abc import Callable  # Type annotation for builder callables

from torch import nn  # PyTorch neural network module base classes

from common.config.types import (
    ModelConfig,  # Strongly typed model configuration dataclass
)
from model.builder import build_timm_model  # High-level model construction function
from model.model_names import (
    ModelArchitecture,  # Enum defining supported architecture keys
)

# Type alias defining signature for model builder functions: ModelConfig -> nn.Module
ModelBuilder = Callable[[ModelConfig], nn.Module]

# Registry table mapping architecture enums to their corresponding construction lambdas
MODEL_REGISTRY = {
    ModelArchitecture.EFFICIENTNET_B0: (
        lambda config: build_timm_model("efficientnet_b0", config)
    )
}
