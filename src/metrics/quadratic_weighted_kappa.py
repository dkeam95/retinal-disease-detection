"""Quadratic Weighted Kappa Metric for ordinal classification."""

from __future__ import annotations  # Enables modern type hints (Python 3.7+)

from sklearn.metrics import cohen_kappa_score  # Scikit-learn Cohen's kappa evaluation metric
from torch import Tensor  # Type annotation for PyTorch multi-dimensional arrays

from metrics._validation import validate_shapes  # Helper function for verifying tensor dimensions


def compute_quadratic_weighted_kappa(
    logits: Tensor,
    targets: Tensor,
) -> float:
    """Compute quadratic weighted kappa (Cohen's kappa with quadratic weighting).

    Args:
        logits:
            Logits of shape (batch_size, num_classes).

        targets:
            Ground truth labels of shape (batch_size,).

    Returns:
        The quadratic weighted kappa score as a float scalar.
    """

    # Validate input tensor shapes and ranks
    validate_shapes(logits, targets)

    # Convert logits to predicted classes along class dimension (dim=1)
    predictions = logits.argmax(dim=1)

    # Compute Cohen's kappa score with quadratic weights on CPU NumPy arrays
    score = cohen_kappa_score(
        y1=targets.cpu().numpy(),
        y2=predictions.cpu().numpy(),
        weights="quadratic",
    )

    # Return quadratic weighted kappa score as float
    return float(score)