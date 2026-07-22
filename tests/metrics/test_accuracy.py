import pytest
import torch

from common.classes import DRClass
from metrics.accuracy import compute_accuracy


def test_perfect_accuracy() -> None:
    """Verify perfect predictions produce accuracy of 1.0."""

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

    accuracy = compute_accuracy(
        logits,
        targets,
    )

    print(f"\nAccuracy = {accuracy:.4f}")

    assert accuracy == pytest.approx(1.0)


def test_partial_accuracy() -> None:
    """Verify partially correct predictions."""

    logits = torch.tensor(
        [
            [10.0, 0.0, 0.0, 0.0, 0.0],  # -> No DR
            [0.0, 10.0, 0.0, 0.0, 0.0],  # -> Mild
            [0.0, 0.0, 0.0, 10.0, 0.0],  # -> Severe (wrong)
            [0.0, 0.0, 0.0, 0.0, 10.0],  # -> Proliferative
        ],
        dtype=torch.float32,
    )

    targets = torch.tensor(
        [
            DRClass.NO_DR,
            DRClass.MILD_NPDR,
            DRClass.MODERATE_NPDR,
            DRClass.PROLIFERATIVE_DR,
        ],
        dtype=torch.long,
    )

    accuracy = compute_accuracy(
        logits,
        targets,
    )

    print(f"\nAccuracy = {accuracy:.4f}")

    assert accuracy == pytest.approx(0.75)


def test_zero_accuracy() -> None:
    """Verify completely incorrect predictions."""

    logits = torch.tensor(
        [
            [10.0, 0.0, 0.0, 0.0, 0.0],  # -> No DR
            [10.0, 0.0, 0.0, 0.0, 0.0],  # -> No DR
            [10.0, 0.0, 0.0, 0.0, 0.0],  # -> No DR
            [10.0, 0.0, 0.0, 0.0, 0.0],  # -> No DR
        ],
        dtype=torch.float32,
    )

    targets = torch.tensor(
        [
            DRClass.MILD_NPDR,
            DRClass.MODERATE_NPDR,
            DRClass.SEVERE_NPDR,
            DRClass.PROLIFERATIVE_DR,
        ],
        dtype=torch.long,
    )

    accuracy = compute_accuracy(
        logits,
        targets,
    )

    print(f"\nAccuracy = {accuracy:.4f}")

    assert accuracy == pytest.approx(0.0)


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
        compute_accuracy(
            logits,
            targets,
        )

    print("\nShape mismatch correctly detected.")