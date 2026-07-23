"""Registry of available model builders.

This module maps model architecture names to their
corresponding builder  functions.
"""

from __future__ import annotations

from collections.abc import Callable

from torch import nn

from common.config.types import ModelConfig
from model.builder import build_timm_model
from model.model_names import ModelArchitecture


ModelBuilder = Callable[[ModelConfig], nn.Module]

MODEL_REGISTRY = {
    ModelArchitecture.EFFICIENTNET_B0:
        lambda config: build_timm_model("efficientnet_b0", config)
}