"""Tests for the model factory."""

from __future__ import annotations

import pytest
from torch import nn

from common.config.types import ModelConfig
from model.factory import create_model


def test_create_efficientnet_b0() -> None:
    """Factory should create an EfficientNet-B0 model."""

    config = ModelConfig(
        architecture="efficientnet_b0" ,
        pretrained=False,
        num_classes=5
    )

    model = create_model(config)

    assert isinstance(model, nn.Module)


def test_unknown_architecture() -> None:
    """Factory should raise ValueError for unknown architecture."""

    config = ModelConfig(
        architecture="unknown",
        pretrained=False,
        num_classes=5
    )
    
    with pytest.raises(ValueError):
        create_model(config)