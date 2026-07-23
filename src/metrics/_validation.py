"""
Shared validation utilities for metrics.
"""

from __future__ import annotations  # Enables modern type hints (Python 3.7+)

from torch import Tensor  # Type annotation for PyTorch multi-dimensional arrays


def validate_shapes(
    logits: Tensor,
    targets: Tensor,
) -> None:
    """
    Validate metric input tensors.

    Args:
        logits:
            Tensor of shape (N, C).

        targets:
            Tensor of shape (N,).

    Raises:
        ValueError:
            If tensor shapes are invalid.
    """

    # Ensure logits tensor is 2D: (batch_size, num_classes)
    if logits.ndim != 2:
        raise ValueError(
            "Logits must have shape (batch_size, num_classes)."
        )

    # Ensure targets tensor is 1D: (batch_size,)
    if targets.ndim != 1:
        raise ValueError(
            "Targets must have shape (batch_size,)."
        )

    # Verify that batch dimensions (dimension 0) of predictions and targets match
    if logits.shape[0] != targets.shape[0]:
        raise ValueError(
            "Batch size mismatch between logits and targets."
        )