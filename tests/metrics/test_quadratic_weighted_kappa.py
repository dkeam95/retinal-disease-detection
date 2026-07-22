import pytest
import torch

from common.classes import DRClass
from metrics.quadratic_weighted_kappa import (
    compute_quadratic_weighted_kappa,
)


def test_perfect_qwk() -> None:
    """Verify perfect predictions produce QWK of 1.0."""

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

    score = compute_quadratic_weighted_kappa(
        logits,
        targets,
    )

    print(f"\nQWK = {score:.4f}")

    assert score == pytest.approx(1.0)


def test_partial_qwk() -> None:
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

    score = compute_quadratic_weighted_kappa(
        logits,
        targets,
    )

    print(f"\nQWK = {score:.4f}")

    assert score == pytest.approx(
        0.9474,
        abs=1e-4
    )


def test_identical_predictions_are_deterministic() -> None:
    """Verify identical inputs always produce identical QWK."""

    logits = torch.randn(
        8,
        5,
    )

    targets = torch.randint(
        0,
        5,
        (8,),
    )

    score1 = compute_quadratic_weighted_kappa(
        logits,
        targets,
    )

    score2 = compute_quadratic_weighted_kappa(
        logits,
        targets,
    )

    print(
        f"\nRun1 = {score1:.4f}"
    )

    print(
        f"Run2 = {score2:.4f}"
    )

    assert score1 == pytest.approx(
        score2,
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
        compute_quadratic_weighted_kappa(
            logits,
            targets,
        )

    print(
        "\nShape mismatch correctly detected."
    )