"""Precision metric for multi-class classification."""

from __future__ import annotations

from torch import Tensor
from sklearn.metrics import precision_score


_SUPPORTED_AVERAGES = {
    "macro",
    "weighted"
}


def _validate_shapes(logits: Tensor, targets: Tensor) -> None:
    """Validate metric input shapes.
    
        Args:
            logits:
                Model output logits of shape (N, C).

            targets:
                Ground-truth labels of shape (N,).

        Raises:
            ValueError:
                If tensor shapes are invalid.
    """

    if logits.ndim != 2:
        raise ValueError(
            "Logits must have shape (batch_size, num_classes)"
        )

    if targets.ndim != 1:
        raise ValueError(
            "Targets must have shape (batch_size,)."
        )

    if logits.shape[0] != targets.shape[0]:
        raise ValueError(
            "Batch size mismatch between logits and targets."
        )

    
def _validate_average(average: str) -> None:
    """Validate averaging strategy.
    
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


def compute_precision(logits: Tensor, targets: Tensor, average: str = "macro") -> float:
    """Compute precision score.
       
        Args:
            logits:
                Model output logits.

            targets:
                Ground-truth labels.

            average:
                Averaging strategy.

        Returns:
            Precision score.     
    """

    _validate_shapes(logits, targets)

    _validate_average(average)

    predictions = logits.argmax(dim=1)

    return float(
        precision_score(
            y_true=targets.cpu().numpy(),
            y_pred=predictions.cpu().numpy(),
            average=average,
            zero_division=0
        )
    )
