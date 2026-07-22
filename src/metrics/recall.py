"""
Recall metric for multi-class classification.
"""

from __future__ import annotations

from torch import Tensor
from sklearn.metrics import recall_score
from metrics._validation import validate_shapes


_SUPPORTED_AVERAGES = {
    "macro",
    "weighted",
}


def _validate_average(
    average: str,
) -> None:
    """
    Validate averaging strategy.

    Args:
        average:
            Averaging strategy.

    Raises:
        ValueError:
            If averaging strategy is unsupported.
    """

    if average not in _SUPPORTED_AVERAGES:
        raise ValueError(
            f"Unsupported averaging strategy: {average}"
        )


def compute_recall(
    logits: Tensor,
    targets: Tensor,
    average: str = "macro",
) -> float:
    """
    Compute recall score.

    Args:
        logits:
            Model output logits.

        targets:
            Ground-truth labels.

        average:
            Averaging strategy.

    Returns:
        Recall score.
    """

    _validate_average(
        average,
    )

    predictions = logits.argmax(
        dim=1,
    )

    return float(
        recall_score(
            y_true=targets.cpu().numpy(),
            y_pred=predictions.cpu().numpy(),
            average=average,
            zero_division=0,
        )
    )