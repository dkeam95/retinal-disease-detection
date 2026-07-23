from losses.loss_names import (
    LossName,
)


def test_loss_names() -> None:
    """Verify supported loss names."""

    assert (
        LossName.CROSS_ENTROPY
        == "cross_entropy"
    )

    assert (
        LossName.WEIGHTED_CROSS_ENTROPY
        == "weighted_cross_entropy"
    )

    assert (
        LossName.FOCAL
        == "focal"
    )

    assert (
        LossName.CLASS_BALANCED_FOCAL
        == "class_balanced_focal"
    )