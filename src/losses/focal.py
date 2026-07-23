"""
Focal Loss implementation.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor
from torch import nn

from common.config.types import LossConfig

from losses.exceptions import (
    LossInitializationError,
)


class FocalLoss(nn.Module):
    """
    Implementation of Focal Loss.

    Reference:
        Lin et al., "Focal Loss for Dense Object Detection"
        https://arxiv.org/abs/1708.02002
    """

    def __init__(
        self,
        gamma: float,
        alpha: float | None,
        reduction: str,
    ) -> None:
        """
        Initialize Focal Loss.

        Args:
            gamma:
                Focusing parameter.

            alpha:
                Class balancing factor.

            reduction:
                Reduction method.

        Raises:
            LossInitializationError:
                If configuration parameters are invalid.
        """

        super().__init__()

        if gamma < 0:
            raise LossInitializationError(
                "Gamma must be non-negative."
            )

        supported_reductions = {
            "mean",
            "sum",
            "none",
        }

        if reduction not in supported_reductions:
            raise LossInitializationError(
                f"Unsupported reduction: "
                f"{reduction}"
            )

        self._gamma = gamma
        self._alpha = alpha
        self._reduction = reduction

    def forward(
        self,
        logits: Tensor,
        targets: Tensor,
    ) -> Tensor:
        """
        Compute Focal Loss.

        Args:
            logits:
                Raw model predictions.

            targets:
                Ground-truth class indices.

        Returns:
            Computed focal loss.
        """

        cross_entropy_loss = F.cross_entropy(
            logits,
            targets,
            reduction="none",
        )

        pt = torch.exp(
            -cross_entropy_loss,
        )

        focal_loss = (
            (1.0 - pt) ** self._gamma
        ) * cross_entropy_loss

        if self._alpha is not None:
            focal_loss = (
                self._alpha
                * focal_loss
            )

        if self._reduction == "mean":
            return focal_loss.mean()

        if self._reduction == "sum":
            return focal_loss.sum()

        return focal_loss


def build_focal_loss(
    config: LossConfig,
    class_weights: Tensor | None = None,
) -> nn.Module:
    """
    Build a Focal Loss instance.

    Args:
        config:
            Loss configuration.

        class_weights:
            Reserved for API compatibility.

    Returns:
        Configured Focal Loss instance.
    """

    return FocalLoss(
        gamma=config.gamma,
        alpha=config.alpha,
        reduction=config.reduction,
    )