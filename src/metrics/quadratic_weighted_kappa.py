"""
Quadratic Weighted Kappa metric for ordinal classification.

This module provides functions to compute Cohen's Quadratic Weighted Kappa (QWK)
score across predicted and target class labels.
"""

from __future__ import annotations  # Enables modern type hints (Python 3.7+)

from sklearn.metrics import cohen_kappa_score
from torch import Tensor

from metrics._validation import (
    validate_shapes,
)


def compute_quadratic_weighted_kappa(
    logits: Tensor,
    targets: Tensor,
) -> float:
    """
    Compute Quadratic Weighted Kappa (QWK) score.

    Args:
        logits:
            Unnormalized model output tensor of shape (N, C).
        targets:
            Ground-truth class labels tensor of shape (N,).

    Returns:
        float:
            Computed Quadratic Weighted Kappa score as a float scalar.
    """

    # Validate input tensor shapes and batch alignment
    validate_shapes(
        logits,
        targets,
    )

    # Extract class index with highest predicted logit along class dimension (dim=1)
    predictions = logits.argmax(
        dim=1,
    )

    # Compute Quadratic Weighted Kappa using scikit-learn
    return float(
        cohen_kappa_score(
            y1=targets.cpu().numpy(),
            y2=predictions.cpu().numpy(),
            weights="quadratic",
        )
    )