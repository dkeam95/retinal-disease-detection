"""Confusion matrix computation for multi-class classification."""

from __future__ import annotations  # Enables modern type hints (Python 3.7+)

import torch  # Core PyTorch library for tensor operations
from sklearn.metrics import confusion_matrix  # Scikit-learn confusion matrix utility
from torch import Tensor  # Type annotation for PyTorch multi-dimensional arrays

from common.classes import DRClass  # Canonical enum containing all class identifiers
from metrics._validation import validate_shapes  # Helper function for verifying tensor dimensions


def compute_confusion_matrix(logits: Tensor, targets: Tensor) -> Tensor:
    """Compute confusion matrix.

    Args:
        logits:
            Model output logits of shape (N, C).

        targets:
            Ground-truth labels of shape (N,).

    Returns:
        Confusion matrix as torch.Tensor of shape (C, C).
    """

    # Validate input tensor ranks and batch size compatibility
    validate_shapes(logits, targets)

    # Extract class index with highest predicted logit along class dimension (dim=1)
    predictions = logits.argmax(dim=1)

    # Calculate confusion matrix using scikit-learn, transferring tensors to CPU NumPy arrays
    matrix = confusion_matrix(
        y_true=targets.cpu().numpy(),
        y_pred=predictions.cpu().numpy(),
        labels=[member.value for member in DRClass],
    )

    # Convert NumPy confusion matrix back to PyTorch int64 tensor
    return torch.tensor(
        matrix,
        dtype=torch.int64,
    )