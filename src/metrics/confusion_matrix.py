"""
Confusion matrix computation for multi-class classification.

This module computes the multi-class confusion matrix, mapping true categorical
disease stages against predicted classes across diabetic retinopathy (DR) categories.
"""

from __future__ import annotations  # Enables modern type hints (Python 3.7+)

import torch
from sklearn.metrics import confusion_matrix
from torch import Tensor

from common.classes import DRClass  # Enum defining Diabetic Retinopathy severity stages
from metrics._validation import (
    validate_shapes,
)


def compute_confusion_matrix(
    logits: Tensor,
    targets: Tensor,
) -> Tensor:
    """
    Compute a multi-class confusion matrix populated across all DR disease stages.

    The resulting matrix $M$ has dimensions $(C, C)$, where rows $i$ represent ground-truth
    classes (true disease stages) and columns $j$ represent model-predicted classes.

    Args:
        logits:
            Unnormalized model predictions tensor of shape (N, C),
            where N is batch size and C is the number of target classes.
        targets:
            Ground-truth categorical class labels tensor of shape (N,).

    Returns:
        Tensor:
            Confusion matrix tensor of shape (C, C) with torch.int64 data type,
            where entry [i, j] counts samples with true label i predicted as class j.
    """

    # Validate input tensor ranks (logits is 2D, targets is 1D) and batch alignment
    validate_shapes(
        logits,
        targets,
    )

    # Convert predicted raw logits to class index via argmax along class dimension (dim=1)
    predictions = logits.argmax(
        dim=1,
    )

    # Compute scikit-learn confusion matrix on CPU NumPy arrays.
    # Explicitly pass `labels` using DRClass enum members to guarantee fixed (C, C) shape
    # even when certain disease classes are absent from the current mini-batch.
    matrix = confusion_matrix(
        y_true=targets.cpu().numpy(),
        y_pred=predictions.cpu().numpy(),
        labels=[member.value for member in DRClass],
    )

    # Convert computed NumPy array back to PyTorch int64 tensor for downstream evaluation
    return torch.tensor(
        matrix,
        dtype=torch.int64,
    )
