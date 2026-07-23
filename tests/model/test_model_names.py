from model.model_names import (
    ModelArchitecture,
)


def test_model_architecture_is_string() -> None:
    """Verify model architecture behaves as string."""

    assert (
        ModelArchitecture.EFFICIENTNET_B0
        == "efficientnet_b0"
    )


def test_model_lookup() -> None:
    """Verify lookup from string."""

    architecture = ModelArchitecture(
        "efficientnet_b0"
    )

    assert (
        architecture
        is ModelArchitecture.EFFICIENTNET_B0
    )