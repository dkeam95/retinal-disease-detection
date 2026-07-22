"""Quadratic Weighted Kappa Metric for ordinal classification."""

from __future__ import annotations

from sklearn.metrics import cohen_kappa_score
from metrics._validation import validate_shapes
from torch import Tensor


def compute_quadratic_weighted_kappa(
    logits: Tensor,
    targets: Tensor,
) -> float:
    """
    Compute quadratic weighted kappa (Cohen's kappa with quadratic weighting).
    
    Args:
        logits: Logits of shape (batch_size, num_classes).
        targets: Ground truth labels of shape (batch_size,).

    Returns:
        The quadratic weighted kappa score.
    """

    # Convert logits to predicted classes
    predictions = logits.argmax(dim=1)

    # Compute Cohen's kappa
    score = cohen_kappa_score(
        y1=targets.cpu().numpy(),
        y2=predictions.cpu().numpy(),
        weights="quadratic"
    )

    return float(score)