from common.config.types import ModelConfig
from model.backbone import build_backbone


def test_build_backbone() -> None:
    """Verify backbone creation."""

    config = ModelConfig(
        architecture="efficientnet_b0",
        pretrained=False,
        num_classes=5,
    )

    backbone = build_backbone(
        "efficientnet_b0",
        config,
    )

    features = backbone.forward_features

    assert callable(
        features,
    )