"""Accuracy metric for multi-class classification.

This module computes standard classification accuracy, representing the ratio of
correctly predicted instances to total samples across multi-class datasets.
"""

from __future__ import annotations     # Enables modern type hints (Python 3.7+)

from torch import Tensor               # Type annotation for PyTorch multi-dimensional arrays

from metrics._validation import validate_shapes  # Import shape validation helper


def compute_accuracy(logits: Tensor, targets: Tensor) -> float:
    """Compute overall multi-class classification accuracy.

    Calculates the proportion of correctly classified instances across all classes
    in the batch.

    Args:
        logits:
            Unnormalized model output tensor of shape (N, C), where N is batch size
            and C is the number of target classes.
        targets:
            Ground-truth categorical class labels tensor of shape (N,).

    Returns:
        float:
            Classification accuracy score as a float scalar in range [0.0, 1.0].
            Returns 0.0 if the input batch contains no samples.
    """

    # Validate input tensor ranks (logits is 2D, targets is 1D) and batch size alignment
    validate_shapes(logits, targets)

    # Extract class index with highest predicted logit along class dimension (dim=1)
    predictions = logits.argmax(dim=1)

    # Count total number of correctly predicted instances as a Python scalar
    correct = (predictions == targets).sum().item()

    # Get total count of samples in the targets batch
    total = targets.numel()

    # Handle empty batch edge case safely to avoid zero-division errors
    if total == 0:
        return 0.0

    # Compute ratio of correct predictions to total samples
    return correct / total