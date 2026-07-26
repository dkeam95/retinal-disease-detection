"""
F1 metric for multi-class classification.

This module provides functions to compute the multi-class F1 score across
predicted and target class labels.
"""

from __future__ import annotations  # Enables modern type hints (Python 3.7+)

from sklearn.metrics import f1_score
from torch import Tensor

from metrics._validation import (
    validate_shapes,
)
from metrics.exceptions import (
    MetricInitializationError,
)

# Supported multi-class aggregation strategies
_SUPPORTED_AVERAGES = {
    "macro",
    "weighted",
}


def _validate_average(
    average: str,
) -> None:
    """
    Validate the requested averaging strategy.

    Args:
        average:
            Averaging strategy identifier.

    Raises:
        MetricInitializationError:
            If the strategy is not supported.
    """

    # Ensure the requested strategy is supported
    if average not in _SUPPORTED_AVERAGES:
        supported = ", ".join(
            sorted(_SUPPORTED_AVERAGES)
        )

        raise MetricInitializationError(
            f"Unsupported averaging strategy: "
            f"{average}. "
            f"Supported values: "
            f"{supported}."
        )


def compute_f1(
    logits: Tensor,
    targets: Tensor,
    average: str = "macro",
) -> float:
    """
    Compute multi-class F1 score.

    Args:
        logits:
            Unnormalized model output tensor of shape (N, C).
        targets:
            Ground-truth class labels tensor of shape (N,).
        average:
            Averaging strategy ("macro" or "weighted"). Defaults to "macro".

    Returns:
        float:
            Computed F1 score as a float scalar.
    """

    # Validate input tensor shapes and batch alignment
    validate_shapes(
        logits,
        targets,
    )

    # Validate averaging strategy
    _validate_average(
        average,
    )

    # Extract class index with highest predicted logit along class dimension (dim=1)
    predictions = logits.argmax(
        dim=1,
    )

    # Compute F1 score using scikit-learn
    return float(
        f1_score(
            y_true=targets.cpu().numpy(),
            y_pred=predictions.cpu().numpy(),
            average=average,
            zero_division=0,
        )
    )