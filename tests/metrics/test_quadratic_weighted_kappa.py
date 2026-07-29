"""
Unit tests for the Quadratic Weighted Kappa (QWK) evaluation metric.

This module contains unit tests verifying the correct computation of the multi-class
QWK score under various conditions, including perfect predictions, partial agreement,
deterministic execution consistency, and input tensor shape validation.
"""

from __future__ import annotations  # Enables modern type hints (Python 3.7+)

import pytest
import torch

from common.classes import DRClass
from metrics.exceptions import MetricInitializationError
from metrics.quadratic_weighted_kappa import (
    compute_quadratic_weighted_kappa,
)


def test_perfect_qwk() -> None:
    """
    Verify that completely correct predictions produce a QWK score of 1.0.
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

    # Compute QWK score
    score = compute_quadratic_weighted_kappa(
        logits,
        targets,
    )

    print(f"\nQWK = {score:.4f}")

    # Assert score is exactly 1.0
    assert score == pytest.approx(1.0)


def test_partial_qwk() -> None:
    """
    Verify that partially correct predictions produce the expected QWK score.
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

    # Compute QWK score on partially correct batch
    score = compute_quadratic_weighted_kappa(
        logits,
        targets,
    )

    print(f"\nQWK = {score:.4f}")

    # Assert score matches expected value within tolerance
    assert score == pytest.approx(
        0.9474,
        abs=1e-4,
    )


def test_identical_predictions_are_deterministic() -> None:
    """
    Verify that identical input tensors always yield identical QWK calculation results.
    """

    # Generate random logits for 8 samples across 5 classes
    logits = torch.randn(
        8,
        5,
    )

    # Generate random target labels
    targets = torch.randint(
        0,
        5,
        (8,),
    )

    # Run first QWK calculation
    score1 = compute_quadratic_weighted_kappa(
        logits,
        targets,
    )

    # Run second QWK calculation with identical inputs
    score2 = compute_quadratic_weighted_kappa(
        logits,
        targets,
    )

    print(f"\nRun1 = {score1:.4f}")
    print(f"Run2 = {score2:.4f}")

    # Assert both computation runs yield identical output
    assert score1 == pytest.approx(score2)


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
        compute_quadratic_weighted_kappa(
            logits,
            targets,
        )

    print("\nShape mismatch correctly detected.")
