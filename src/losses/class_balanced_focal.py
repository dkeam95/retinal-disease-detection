"""Class-Balanced Focal Loss implementation."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor
from torch import nn

from common.config.types import LossConfig


class ClassBalancedFocalLoss(nn.Module):
    """Implementation of Class-Balanced Focal Loss."""

    _class_weights: Tensor

    def __init__(
        self,
        gamma: float,
        reduction: str,
        class_weights: Tensor,
    ) -> None:
        """Initialize Class-Balanced Focal Loss.

        Args:
            gamma:
                Focusing parameter.

            reduction:
                Reduction method.

            class_weights:
                Precomputed class weights.
        """

        super().__init__()

        self._gamma = gamma
        self._reduction = reduction

        self.register_buffer(
            "_class_weights",
            class_weights,
        )

    def forward(
        self,
        logits: Tensor,
        targets: Tensor,
    ) -> Tensor:
        """Compute Class-Balanced Focal Loss.

        Args:
            logits:
                Model predictions before softmax.

            targets:
                Ground-truth class indices.

        Returns:
            Computed loss.
        """

        cross_entropy_loss = F.cross_entropy(
            logits,
            targets,
            weight=self._class_weights,
            reduction="none",
        )

        pt = torch.exp(-cross_entropy_loss)

        focal_loss = (
            (1.0 - pt) ** self._gamma
        ) * cross_entropy_loss

        if self._reduction == "mean":
            return focal_loss.mean()

        if self._reduction == "sum":
            return focal_loss.sum()

        if self._reduction == "none":
            return focal_loss

        raise ValueError(
            f"Unsupported reduction: {self._reduction}",
        )


def build_class_balanced_focal_loss(
    config: LossConfig,
    class_weights: Tensor | None = None,
) -> nn.Module:
    """Build a Class-Balanced Focal Loss instance.

    Args:
        config:
            Loss configuration.

        class_weights:
            Precomputed class weights.

    Returns:
        Configured Class-Balanced Focal Loss instance.
    """

    if class_weights is None:
        raise ValueError(
            "Class weights must be provided for class balanced focal loss."
        )

    return ClassBalancedFocalLoss(
        gamma=config.gamma,
        reduction=config.reduction,
        class_weights=class_weights,
    )