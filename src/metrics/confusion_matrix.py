"""Confusion matrix computation for multi-class classification."""

from __future__ import annotations

from common.classes import DRClass

import torch
from sklearn.metrics import confusion_matrix
from metrics._validation import validate_shapes
from torch import Tensor


def compute_confusion_matrix(logits: Tensor, targets: Tensor) -> Tensor:
    """Compute confusion matrix.
    
       Args:
        logits:
            Model output logits.

        targets:
            Ground-truth labels.

       Returns:
            Confusion matrix as torch.Tensor.
    """

    predictions = logits.argmax(dim=1)

    matrix = confusion_matrix(
        y_true=targets.cpu().numpy(),
        y_pred=predictions.cpu().numpy(),
        labels=[member.value for member in DRClass]
    )

    return torch.tensor(
        matrix,
        dtype=torch.int64
    )
    
    
    
    