"""Model registry."""

from __future__ import annotations

from collections.abc import Callable

from torch import nn

from common.config.types import ModelConfig
from model.builders import build_efficientnet_b0


ModelBuilder = Callable[[ModelConfig], nn.Module]

MODEL_REGISTRY: dict[str, ModelBuilder] = {
    "efficientnet_b0": build_efficientnet_b0,
}