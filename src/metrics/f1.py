"""
F1 metric for multi-class classification.
"""

from __future__ import annotations

from sklearn.metrics import f1_score
from metrics._validation import validate_shapes
from torch import Tensor


_SUPPORTED_AVERAGES = {
    "macro",
    "weighted",
}


def _validate_average(
    average: str,
) -> None:
    """Validate averaging strategy."""

    if average not in _SUPPORTED_AVERAGES:
        raise ValueError(
            f"Unsupported averaging strategy: {average}"
        )


def compute_f1(
    logits: Tensor,
    targets: Tensor,
    average: str = "macro",
) -> float:
    """
    Compute F1 score.
    """

    _validate_average(
        average,
    )

    predictions = logits.argmax(
        dim=1,
    )

    return float(
        f1_score(
            y_true=targets.cpu().numpy(),
            y_pred=predictions.cpu().numpy(),
            average=average,
            zero_division=0,
        )
    )