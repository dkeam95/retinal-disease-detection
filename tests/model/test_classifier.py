import torch

from common.config.types import ModelConfig
from model.classifier import build_classifier


def test_build_classifier() -> None:
    """Verify classifier creation."""

    config = ModelConfig(
        architecture="efficientnet_b0",
        pretrained=False,
        num_classes=5,
    )

    classifier = build_classifier(
        in_features=1280,
        config=config,
    )

    x = torch.randn(
        4,
        1280,
    )

    logits = classifier(
        x,
    )

    assert logits.shape == (
        4,
        5,
    )

    print(logits.shape)