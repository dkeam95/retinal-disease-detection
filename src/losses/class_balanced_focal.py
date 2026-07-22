"""Class-Balanced Focal Loss implementation."""

from __future__ import annotations   # Enables modern type hints (Python 3.7+)

import torch
import torch.nn.functional as F      # Functional interface for PyTorch operations
from torch import Tensor
from torch import nn                 # Neural network base module

from common.config.types import LossConfig  # Configuration object holding loss hyperparameters


class ClassBalancedFocalLoss(nn.Module):
    """Implementation of Class-Balanced Focal Loss."""

    _class_weights: Tensor  # Explicit attribute type declaration for registered PyTorch buffer

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

        # Register class_weights as a persistent buffer so it automatically moves to the correct GPU/CPU device with the module
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

        # Compute unreduced Cross Entropy loss weighted by class frequencies
        cross_entropy_loss = F.cross_entropy(
            logits,
            targets,
            weight=self._class_weights,
            reduction="none",
        )

        # Estimate class probability p_t = exp(-CE_loss)
        pt = torch.exp(-cross_entropy_loss)

        # Apply focal modulating factor (1 - p_t)^gamma to down-weight well-classified samples
        focal_loss = (
            (1.0 - pt) ** self._gamma
        ) * cross_entropy_loss

        # Apply requested loss reduction across batch dimensions
        if self._reduction == "mean":
            return focal_loss.mean()

        if self._reduction == "sum":
            return focal_loss.sum()

        if self._reduction == "none":
            return focal_loss

        # Raise exception for unsupported reduction parameters
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

    # Validate that precomputed class weights are passed for class-balanced mode
    if class_weights is None:
        raise ValueError(
            "Class weights must be provided for class balanced focal loss."
        )

    # Instantiate ClassBalancedFocalLoss module with parameters from config and weights tensor
    return ClassBalancedFocalLoss(
        gamma=config.gamma,
        reduction=config.reduction,
        class_weights=class_weights,
    )