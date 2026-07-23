"""Accuracy metric for multi-class classification."""

from __future__ import annotations     # Enables modern type hints (Python 3.7+)

from torch import Tensor               # Type annotation for PyTorch multi-dimensional arrays

from metrics._validation import validate_shapes  # Import shape validation helper


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

    # Validate input tensor ranks and batch size compatibility
    validate_shapes(logits, targets)

    # Extract class index with highest predicted logit along class dimension (dim=1)
    predictions = logits.argmax(dim=1)

    # Count total number of correctly predicted instances as a Python scalar
    correct = (predictions == targets).sum().item()

    # Get total count of samples in the targets batch
    total = targets.numel()

    # Compute ratio of correct predictions to total samples
    return correct / total