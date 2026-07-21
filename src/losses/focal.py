"""Focal Loss implementation."""

from __future__ import annotations  # Enables modern type hints (Python 3.7+)

import torch                        # Core PyTorch library for tensor computations
import torch.nn.functional as F     # Functional interface for loss functions and activations
from torch import Tensor            # Type annotation for PyTorch multi-dimensional arrays
from torch import nn                # Neural network modules base class

from common.config.types import LossConfig  # Configuration object holding loss hyperparameters


class FocalLoss(nn.Module):
    """Implementation of Focal Loss."""

    def __init__(
        self,
        gamma: float,
        alpha: float | None,
        reduction: str,
    ) -> None:
        """Initialize Focal Loss.

        Args:
            gamma:
                Focusing parameter.

            alpha:
                Class balancing factor.

            reduction:
                Reduction method.
        """

        super().__init__()

        # Focusing parameter (gamma >= 0) that down-weights easy examples
        self._gamma = gamma
        # Optional weighting factor (alpha) to address class imbalance
        self._alpha = alpha
        # Loss reduction strategy ('mean', 'sum', or 'none')
        self._reduction = reduction

    def forward(
        self,
        logits: Tensor,
        targets: Tensor,
    ) -> Tensor:
        """Compute Focal Loss.

        Args:
            logits:
                Model predictions before softmax.

            targets:
                Ground-truth class indices.

        Returns:
            Computed loss.
        """

        # Compute per-sample unreduced cross entropy loss: CE(p_t) = -log(p_t)
        cross_entropy_loss = F.cross_entropy(
            logits,
            targets,
            reduction="none",
        )

        # Reconstruct model probability for the correct target class: p_t = exp(-CE)
        pt = torch.exp(-cross_entropy_loss)

        # Apply focal weighting term: FL(p_t) = (1 - p_t)^gamma * CE(p_t)
        focal_loss = (
            (1.0 - pt) ** self._gamma
        ) * cross_entropy_loss

        # Scale by class balancing parameter alpha if specified
        if self._alpha is not None:
            focal_loss = self._alpha * focal_loss

        # Apply requested reduction across the batch
        if self._reduction == "mean":
            return focal_loss.mean()

        if self._reduction == "sum":
            return focal_loss.sum()

        if self._reduction == "none":
            return focal_loss

        raise ValueError(
            f"Unsupported reduction: {self._reduction}"
        )


def build_focal_loss(
    config: LossConfig,
    class_weights: Tensor | None = None,
) -> nn.Module:
    """Build a Focal Loss instance.

    Args:
        config:
            Loss configuration.

    Returns:
        Configured Focal Loss instance.
    """

    # Instantiate FocalLoss using parameters extracted from the configuration schema
    return FocalLoss(
        gamma=config.gamma,
        alpha=config.alpha,
        reduction=config.reduction,
    )