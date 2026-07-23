"""
F1 metric for multi-class classification.
"""

from __future__ import annotations  # Enables modern type hints (Python 3.7+)

from sklearn.metrics import f1_score  # Scikit-learn F1-score evaluation metric
from torch import Tensor  # Type annotation for PyTorch multi-dimensional arrays

from metrics._validation import validate_shapes  # Helper function for verifying tensor dimensions

# Set of allowed multi-class averaging methods ('macro' or 'weighted')
_SUPPORTED_AVERAGES = {
    "macro",
    "weighted",
}


def _validate_average(
    average: str,
) -> None:
    """Validate averaging strategy."""

    # Ensure requested averaging strategy is within supported options
    if average not in _SUPPORTED_AVERAGES:
        raise ValueError(
            f"Unsupported averaging strategy: {average}"
        )


def compute_f1(
    logits: Tensor,
    targets: Tensor,
    average: str = "macro",
) -> float:
    """Compute F1 score.

    Args:
        logits:
            Model output logits of shape (N, C).

        targets:
            Ground-truth labels of shape (N,).

        average:
            Averaging strategy ('macro' or 'weighted'). Defaults to 'macro'.

    Returns:
        Computed F1 score as a float scalar.
    """

    # Validate input tensor shapes and ranks
    validate_shapes(logits, targets)

    # Validate averaging mode parameter
    _validate_average(
        average,
    )

    # Extract class index with highest predicted logit along class dimension (dim=1)
    predictions = logits.argmax(
        dim=1,
    )

    # Calculate F1 score using scikit-learn on CPU NumPy arrays, suppressing zero-division warnings
    return float(
        f1_score(
            y_true=targets.cpu().numpy(),
            y_pred=predictions.cpu().numpy(),
            average=average,
            zero_division=0,
        )
    )