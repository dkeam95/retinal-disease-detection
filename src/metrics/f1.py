"""
F1 metric for multi-class classification.
"""

from __future__ import annotations

from sklearn.metrics import f1_score
from torch import Tensor


_SUPPORTED_AVERAGES = {
    "macro",
    "weighted",
}


def _validate_shapes(
    logits: Tensor,
    targets: Tensor,
) -> None:
    """Validate metric input shapes."""

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

    _validate_shapes(
        logits,
        targets,
    )

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