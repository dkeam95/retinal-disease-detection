from common.config.types import ModelConfig
from model.builder import build_timm_model


def test_build_timm_model() -> None:
    """Verify TIMM model creation."""

    config = ModelConfig(
        architecture="efficientnet_b0",
        pretrained=False,
        num_classes=5,
    )

    model = build_timm_model(
        "efficientnet_b0",
        config,
    )

    assert model is not None

    print(model.__class__.__name__)