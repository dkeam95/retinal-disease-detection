import pytest
import torch

from common.classes import DRClass
from metrics.f1 import compute_f1


def test_perfect_f1() -> None:
    """Verify perfect predictions produce F1 of 1.0."""

    logits = torch.tensor(
        [
            [10.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 10.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 10.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 10.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 10.0],
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

    score = compute_f1(
        logits,
        targets,
    )

    print(f"\nF1 = {score:.4f}")

    assert score == pytest.approx(1.0)


def test_partial_f1() -> None:
    """Verify partially correct predictions."""

    logits = torch.tensor(
        [
            [10.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 10.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 10.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 10.0],
            [0.0, 0.0, 10.0, 0.0, 0.0],
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

    score = compute_f1(
        logits,
        targets,
    )

    print(f"\nF1 = {score:.4f}")

    assert score == pytest.approx(
        0.7333333333333333,
        abs=1e-2
    )


def test_invalid_average() -> None:
    """Verify unsupported averaging strategy."""

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
        compute_f1(
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
        compute_f1(
            logits,
            targets,
        )

    print(
        "\nShape mismatch correctly detected."
    )