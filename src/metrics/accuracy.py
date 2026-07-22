"""Accuracy metric for multi-class classification."""

from __future__ import annotations

from torch import Tensor


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


def compute_accuracy(logits: Tensor, targets: Tensor) -> float:
    """Compute classification accuracy.
    
    Args:
        logits:
            Model output logits of shape (N, C).
        targets:
            Ground-truth labels of shape (N,).

    Returns:
        Classification accuracy.
    """

    _validate_shapes(logits, targets)

    predictions = logits.argmax(dim=1)

    correct = (predictions == targets).sum().item()

    total = targets.numel()
    
    return correct / total

    