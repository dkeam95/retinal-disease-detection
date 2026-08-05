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
    ),
    ModelArchitecture.RESNET18: (
        lambda config: build_timm_model("resnet18", config)
    ),
    ModelArchitecture.RESNET50: (
        lambda config: build_timm_model("resnet50", config)
    ),
    ModelArchitecture.DENSENET121: (
        lambda config: build_timm_model("densenet121", config)
    ),
    ModelArchitecture.MOBILENET_V3_LARGE: (
        lambda config: build_timm_model("mobilenetv3_large_100", config)
    ),
    ModelArchitecture.CONVNEXT_TINY: (
        lambda config: build_timm_model("convnext_tiny", config)
    ),
    ModelArchitecture.VIT_B_16: (
        lambda config: build_timm_model("vit_base_patch16_224", config)
    ),
    ModelArchitecture.SWIN_T: (
        lambda config: build_timm_model("swin_tiny_patch4_window7_224", config)
    ),
}
