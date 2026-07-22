import pytest
import torch

from common.classes import DRClass
from metrics.precision import compute_precision


def test_perfect_precision() -> None:
    """Verify perfect predictions produce precision of 1.0."""

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

    precision = compute_precision(
        logits,
        targets,
    )

    print(f"\nPrecision = {precision:.4f}")

    assert precision == pytest.approx(1.0)


def test_partial_precision() -> None:
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

    precision = compute_precision(
        logits,
        targets,
    )

    print(f"\nPrecision = {precision:.4f}")

    assert precision == pytest.approx(
        0.80
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
        compute_precision(
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
        compute_precision(
            logits,
            targets,
        )

    print(
        "\nShape mismatch correctly detected."
    )