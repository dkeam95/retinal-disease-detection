"""
Shared validation utilities for evaluation metrics.

This module provides reusable validation helper functions to ensure that input
tensors (logits and targets) satisfy structural constraints (rank, dimensionality,
and batch alignment) before calculating performance metrics.
"""

from __future__ import annotations  # Enables modern type hints (Python 3.7+)

from torch import Tensor  # Type annotation for PyTorch multi-dimensional arrays

from metrics.exceptions import (
    MetricInitializationError,
)


def validate_shapes(
    logits: Tensor,
    targets: Tensor,
) -> None:
    """
    Validate metric input tensor ranks and batch dimension compatibility.

    Args:
        logits:
            Unnormalized model output predictions tensor of shape (N, C),
            where N is batch size and C is the number of target classes.
        targets:
            Ground-truth categorical class labels tensor of shape (N,).

    Raises:
        MetricInitializationError:
            If logits tensor is not 2D (rank 2), targets tensor is not 1D (rank 1),
            or if the batch dimensions (N) of logits and targets do not match.
    """

    # Verify that predicted logits tensor is 2-dimensional (batch_size, num_classes)
    if logits.ndim != 2:
        raise MetricInitializationError(
            "Logits must have shape "
            "(batch_size, num_classes)."
        )

    # Verify that target labels tensor is 1-dimensional (batch_size,)
    if targets.ndim != 1:
        raise MetricInitializationError(
            "Targets must have shape "
            "(batch_size,)."
        )

    # Ensure batch sizes (dimension index 0) between predictions and targets are identical
    if logits.shape[0] != targets.shape[0]:
        raise MetricInitializationError(
            "Batch size mismatch between "
            "logits and targets."
        )
