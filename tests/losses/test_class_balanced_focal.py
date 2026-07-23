import pytest
import torch

from common.config.types import LossConfig

from losses.class_balanced_focal import (
    build_class_balanced_focal_loss,
)

from losses.exceptions import (
    LossInitializationError,
)


def test_invalid_reduction() -> None:
    """Verify invalid reduction raises LossInitializationError."""

    class_weights = torch.ones(
        5,
        dtype=torch.float32,
    )

    config = LossConfig(
        name="class_balanced_focal",
        gamma=2.0,
        reduction="invalid",
    )

    with pytest.raises(
        LossInitializationError,
    ):
        build_class_balanced_focal_loss(
            config,
            class_weights,
        )

    print(
        "\nLossInitializationError successfully raised."
    )