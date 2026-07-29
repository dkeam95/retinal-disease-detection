"""
Unit tests for the Confusion Matrix metric computation.

This module contains unit tests verifying the correct generation of multi-class
confusion matrices, including checks for perfect predictions, misclassifications,
fixed matrix dimensions, and input tensor shape validation.
"""

from __future__ import annotations  # Enables modern type hints (Python 3.7+)

import pytest
import torch

from common.classes import DRClass
from metrics.confusion_matrix import (
    compute_confusion_matrix,
)
from metrics.exceptions import MetricInitializationError


def test_perfect_confusion_matrix() -> None:
    """
    Verify that perfect predictions produce an identity confusion matrix.
    """

    # Prepare logits where each sample precisely predicts its true target class
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

    # Set matching targets covering all diabetic retinopathy classes
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

    # Compute confusion matrix
    matrix = compute_confusion_matrix(
        logits,
        targets,
    )

    print("\nConfusion Matrix:")
    print(matrix)

    # Expected result for 100% accuracy is a 5x5 identity matrix
    expected = torch.eye(
        5,
        dtype=torch.int64,
    )

    # Assert matrix matches the identity matrix exactly
    assert torch.equal(
        matrix,
        expected,
    )


def test_partial_confusion_matrix() -> None:
    """
    Verify matrix generation with mixed correct and incorrect predictions.
    """

    # Prepare logits with a mix of correct predictions and intentional misclassifications
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

    # Set ground-truth targets for the batch
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

    # Compute confusion matrix
    matrix = compute_confusion_matrix(
        logits,
        targets,
    )

    print("\nConfusion Matrix:")
    print(matrix)

    # Assert shape is maintained as 5x5 across classes
    assert matrix.shape == (
        5,
        5,
    )

    # Assert total elements counted equal total input samples
    assert matrix.sum().item() == 5


def test_matrix_shape() -> None:
    """
    Verify that the output confusion matrix consistently has shape (C, C).
    """

    # Random logits for a batch of 16 samples across 5 classes
    logits = torch.randn(
        16,
        5,
    )

    # Random targets for 16 samples
    targets = torch.randint(
        0,
        5,
        (16,),
    )

    # Compute confusion matrix
    matrix = compute_confusion_matrix(
        logits,
        targets,
    )

    print(f"\nMatrix shape = {matrix.shape}")

    # Assert shape is strictly (5, 5) corresponding to DRClass count
    assert matrix.shape == (
        5,
        5,
    )


def test_shape_mismatch() -> None:
    """
    Verify that mismatched batch sizes between logits and targets raise MetricInitializationError.
    """

    # Logits with batch size of 4
    logits = torch.randn(
        4,
        5,
    )

    # Targets with mismatched batch size of 3
    targets = torch.tensor(
        [
            DRClass.NO_DR,
            DRClass.MILD_NPDR,
            DRClass.MODERATE_NPDR,
        ],
        dtype=torch.long,
    )

    # Assert that MetricInitializationError is raised due to batch size mismatch
    with pytest.raises(
        MetricInitializationError,
        match="Batch size mismatch",
    ):
        compute_confusion_matrix(
            logits,
            targets,
        )

    print(
        "\nShape mismatch correctly detected."
    )
