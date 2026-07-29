"""
Unit tests for the Accuracy evaluation metric.

This module contains unit tests verifying the correct computation of classification
accuracy under various conditions, including perfect accuracy, partial accuracy,
zero accuracy, and input shape validation.
"""

from __future__ import annotations  # Enables modern type hints (Python 3.7+)

import pytest
import torch

from common.classes import DRClass
from metrics.accuracy import compute_accuracy
from metrics.exceptions import MetricInitializationError


def test_perfect_accuracy() -> None:
    """
    Verify that completely correct predictions produce an accuracy score of 1.0.
    """

    # Prepare logits where the highest value for each sample corresponds to the true class
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

    # Set matching targets for each diabetic retinopathy severity stage
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

    # Compute classification accuracy
    accuracy = compute_accuracy(
        logits,
        targets,
    )

    print(f"\nAccuracy = {accuracy:.4f}")

    # Assert accuracy is exactly 1.0
    assert accuracy == pytest.approx(
        1.0,
    )


def test_partial_accuracy() -> None:
    """
    Verify that a mix of correct and incorrect predictions yields the expected fractional score.
    """

    # Prepare logits resulting in 3 correct predictions out of 4 total samples (0.75 accuracy)
    logits = torch.tensor(
        [
            [10.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 10.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 10.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 10.0],
        ],
        dtype=torch.float32,
    )

    # Ground-truth targets (index 2 mispredicts SEVERE_NPDR instead of MODERATE_NPDR)
    targets = torch.tensor(
        [
            DRClass.NO_DR,
            DRClass.MILD_NPDR,
            DRClass.MODERATE_NPDR,
            DRClass.PROLIFERATIVE_DR,
        ],
        dtype=torch.long,
    )

    # Compute classification accuracy
    accuracy = compute_accuracy(
        logits,
        targets,
    )

    print(f"\nAccuracy = {accuracy:.4f}")

    # Assert accuracy matches expected 0.75 ratio
    assert accuracy == pytest.approx(
        0.75,
    )


def test_zero_accuracy() -> None:
    """
    Verify that completely incorrect predictions produce an accuracy score of 0.0.
    """

    # Prepare logits predicting NO_DR (index 0) for all samples
    logits = torch.tensor(
        [
            [10.0, 0.0, 0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )

    # Ground-truth targets containing only non-zero disease classes
    targets = torch.tensor(
        [
            DRClass.MILD_NPDR,
            DRClass.MODERATE_NPDR,
            DRClass.SEVERE_NPDR,
            DRClass.PROLIFERATIVE_DR,
        ],
        dtype=torch.long,
    )

    # Compute classification accuracy
    accuracy = compute_accuracy(
        logits,
        targets,
    )

    print(f"\nAccuracy = {accuracy:.4f}")

    # Assert accuracy is exactly 0.0
    assert accuracy == pytest.approx(
        0.0,
    )


def test_shape_mismatch() -> None:
    """
    Verify that mismatched batch dimensions between logits and targets raise MetricInitializationError.
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
        compute_accuracy(
            logits,
            targets,
        )

    print(
        "\nShape mismatch correctly detected."
    )
