import torch

from model.types import ModelOutput


def test_model_output() -> None:
    """Verify model output container."""

    logits = torch.randn(
        4,
        5,
    )

    features = torch.randn(
        4,
        1280,
    )

    output = ModelOutput(
        logits=logits,
        features=features,
    )

    assert output.logits.shape == (
        4,
        5,
    )

    assert output.features.shape == (
        4,
        1280,
    )

    print(output)