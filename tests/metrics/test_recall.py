import pytest
import torch

from common.classes import DRClass
from metrics.recall import compute_recall


def test_perfect_recall() -> None:
    """Verify perfect predictions produce recall of 1.0."""

    logits = torch.tensor(
        [
            [10.0, 0.0, 0.0, 0.0, 0.0],  # -> No DR
            [0.0, 10.0, 0.0, 0.0, 0.0],  # -> Mild
            [0.0, 0.0, 10.0, 0.0, 0.0],  # -> Moderate
            [0.0, 0.0, 0.0, 10.0, 0.0],  # -> Severe
            [0.0, 0.0, 0.0, 0.0, 10.0],  # -> Proliferative
        ],
        dtype=torch.float32,
    )

    targets = torch.tensor(
        [
            DRClass.NO_DR,
            DRClass.MILD_NPDR,
            DRClass.MODERATE_NPDR,
            DRClass.SEVERE_NPDR,
            DRClass.PROLIFERATIVE_DR,
        ],
        dtype=torch.long,
    )

    recall = compute_recall(
        logits,
        targets,
    )

    print(f"\nRecall = {recall:.4f}")

    assert recall == pytest.approx(1.0)


def test_partial_recall() -> None:
    """Verify partially correct predictions."""

    logits = torch.tensor(
        [
            [10.0, 0.0, 0.0, 0.0, 0.0],  # -> No DR
            [0.0, 10.0, 0.0, 0.0, 0.0],  # -> Mild
            [0.0, 0.0, 0.0, 10.0, 0.0],  # -> Severe (wrong)
            [0.0, 0.0, 0.0, 0.0, 10.0],  # -> Proliferative
            [0.0, 0.0, 10.0, 0.0, 0.0],  # -> Moderate
        ],
        dtype=torch.float32,
    )

    targets = torch.tensor(
        [
            DRClass.NO_DR,
            DRClass.MILD_NPDR,
            DRClass.MODERATE_NPDR,
            DRClass.PROLIFERATIVE_DR,
            DRClass.MODERATE_NPDR,
        ],
        dtype=torch.long,
    )

    recall = compute_recall(
        logits,
        targets,
    )

    print(f"\nRecall = {recall:.4f}")

    assert recall == pytest.approx(
        0.7333333333333333,
        abs=1e-2
    )


def test_invalid_average() -> None:
    """Verify unsupported averaging strategy raises ValueError."""

    logits = torch.randn(
        4,
        5,
    )

    targets = torch.tensor(
        [
            DRClass.NO_DR,
            DRClass.MILD_NPDR,
            DRClass.MODERATE_NPDR,
            DRClass.SEVERE_NPDR,
        ],
        dtype=torch.long,
    )

    with pytest.raises(
        ValueError,
        match="Unsupported averaging strategy",
    ):
        compute_recall(
            logits,
            targets,
            average="binary",
        )

    print(
        "\nUnsupported averaging strategy correctly detected."
    )


def test_shape_mismatch() -> None:
    """Verify shape validation."""

    logits = torch.randn(
        4,
        5,
    )

    targets = torch.tensor(
        [
            DRClass.NO_DR,
            DRClass.MILD_NPDR,
            DRClass.MODERATE_NPDR,
        ],
        dtype=torch.long,
    )

    with pytest.raises(
        ValueError,
        match="Batch size mismatch",
    ):
        compute_recall(
            logits,
            targets,
        )

    print(
        "\nShape mismatch correctly detected."
    )