"""
Shared validation utilities for metrics.
"""

from __future__ import annotations

from torch import Tensor


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