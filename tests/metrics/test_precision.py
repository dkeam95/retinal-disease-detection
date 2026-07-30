"""
Unit tests for the Precision evaluation metric.

This module contains unit tests verifying the correct computation of the multi-class
Precision score under various conditions, including perfect predictions, partial agreement,
unsupported averaging strategy handling, and input tensor shape validation.
"""

from __future__ import annotations  # Enables modern type hints (Python 3.7+)

import pytest
import torch

from common.classes import DRClass
from metrics.exceptions import MetricInitializationError
from metrics.precision import compute_precision


def test_perfect_precision() -> None:
    """
    Verify that completely correct predictions produce a Precision score of 1.0.
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

    # Compute macro Precision score
    precision = compute_precision(
        logits,
        targets,
    )

    print(f"\nPrecision = {precision:.4f}")

    # Assert score is exactly 1.0
    assert precision == pytest.approx(
        1.0,
    )


def test_partial_precision() -> None:
    """
    Verify that partially correct predictions produce the expected macro Precision score.
    """

    # Define logits with partial misclassifications across batch
    logits = torch.tensor(
        [
            [10.0, 0.0, 0.0, 0.0, 0.0],  # Predicts NO_DR
            [0.0, 10.0, 0.0, 0.0, 0.0],  # Predicts MILD_NPDR
            [0.0, 0.0, 0.0, 10.0, 0.0],  # Predicts SEVERE_NPDR
            [0.0, 0.0, 0.0, 0.0, 10.0],  # Predicts PROLIFERATIVE_DR
            [0.0, 0.0, 10.0, 0.0, 0.0],  # Predicts MODERATE_NPDR
        ],
        dtype=torch.float32,
    )

    # Define ground truth targets for partial match
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

    # Compute Precision score on partially correct batch
    precision = compute_precision(
        logits,
        targets,
    )

    print(f"\nPrecision = {precision:.4f}")

    # Assert score matches expected value within tolerance
    assert precision == pytest.approx(
        0.80,
    )


def test_invalid_average() -> None:
    """
    Verify that passing an unsupported averaging strategy raises MetricInitializationError.
    """

    # Dummy logits for 4 samples across 5 classes
    logits = torch.randn(
        4,
        5,
    )

    # Dummy target labels
    targets = torch.tensor(
        [
            DRClass.NO_DR,
            DRClass.MILD_NPDR,
            DRClass.MODERATE_NPDR,
            DRClass.SEVERE_NPDR,
        ],
        dtype=torch.long,
    )

    # Assert that MetricInitializationError is raised for invalid average parameter
    with pytest.raises(
        MetricInitializationError,
        match="Unsupported averaging strategy",
    ):
        compute_precision(
            logits,
            targets,
            average="binary",
        )

    print("\nUnsupported averaging strategy correctly detected.")


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
        compute_precision(
            logits,
            targets,
        )

    print("\nShape mismatch correctly detected.")
