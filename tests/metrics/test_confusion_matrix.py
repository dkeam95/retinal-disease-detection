import pytest
import torch

from common.classes import DRClass
from metrics.confusion_matrix import (
    compute_confusion_matrix,
)


def test_perfect_confusion_matrix() -> None:
    """Verify perfect confusion matrix."""

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

    matrix = compute_confusion_matrix(
        logits,
        targets,
    )

    print("\nConfusion Matrix:")
    print(matrix)

    expected = torch.eye(
        5,
        dtype=torch.int64,
    )

    assert torch.equal(
        matrix,
        expected,
    )


def test_partial_confusion_matrix() -> None:
    """Verify partially correct confusion matrix."""

    logits = torch.tensor(
        [
            [10.0, 0.0, 0.0, 0.0, 0.0],  # No DR
            [0.0, 10.0, 0.0, 0.0, 0.0],  # Mild
            [0.0, 0.0, 0.0, 10.0, 0.0],  # Severe
            [0.0, 0.0, 0.0, 0.0, 10.0],  # Proliferative
            [0.0, 0.0, 10.0, 0.0, 0.0],  # Moderate
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

    matrix = compute_confusion_matrix(
        logits,
        targets,
    )

    print("\nConfusion Matrix:")
    print(matrix)

    assert matrix.shape == (5, 5)

    assert matrix.sum().item() == 5


def test_matrix_shape() -> None:
    """Verify matrix always has expected dimensions."""

    logits = torch.randn(
        16,
        5,
    )

    targets = torch.randint(
        0,
        5,
        (16,),
    )

    matrix = compute_confusion_matrix(
        logits,
        targets,
    )

    print(f"\nMatrix shape = {matrix.shape}")

    assert matrix.shape == (
        5,
        5,
    )


def test_shape_mismatch() -> None:
    """Verify invalid batch sizes."""

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
        compute_confusion_matrix(
            logits,
            targets,
        )

    print(
        "\nShape mismatch correctly detected."
    )