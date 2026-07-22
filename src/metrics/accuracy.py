"""Accuracy metric for multi-class classification."""

from __future__ import annotations

from torch import Tensor

from metrics._validation import validate_shapes


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

    predictions = logits.argmax(dim=1)

    correct = (predictions == targets).sum().item()

    total = targets.numel()
    
    return correct / total

    