"""Quadratic Weighted Kappa Metric for ordinal classification."""

from __future__ import annotations

from sklearn.metrics import cohen_kappa_score
from torch import Tensor


def _validate_shapes(logits: Tensor, targets: Tensor) -> None:
    """Validate metrics input shapes.
    
    Args:
        logits: Model output logits of shape (N, C).
        targets: Ground-truth labels of shape (N,).

    Raises:
        ValueError:
            If tensor shapes are invalid.
    """

    if logits.ndim != 2:
        raise ValueError(
            "Logits must have shape (batch_size, num_classes)."
        )

    if targets.ndim != 1:
        raise ValueError(
            "Targets must have shape (batch_size,)."
        )

    if logits.shape[0] != targets.shape[0]:
        raise ValueError(
            "Batch size mismatch between logits and targets."
        )


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
    _validate_shapes(logits, targets)

    # Convert logits to predicted classes
    predictions = logits.argmax(dim=1)

    # Compute Cohen's kappa
    score = cohen_kappa_score(
        y1=targets.cpu().numpy(),
        y2=predictions.cpu().numpy(),
        weights="quadratic"
    )

    return float(score)