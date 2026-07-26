"""
Class-Balanced Focal Loss implementation.

This loss function is a variation of focal loss that is designed to address
class imbalance in the dataset.

"""

from __future__ import annotations              # Enables modern type hints (Python 3.7+)

import torch                                    # PyTorch tensor library
import torch.nn.functional as F                 # Functional interface for standard loss routines
from torch import Tensor                        # Type annotation for PyTorch Tensors
from torch import nn                            # PyTorch base module class

from common.config.types import LossConfig      # Configuration dataclass for loss parameters

from losses.exceptions import (                  # Exception raised when loss setup fails
    LossInitializationError,
)


class ClassBalancedFocalLoss(nn.Module):
    """
    Implementation of Class-Balanced Focal Loss.

    """

    _class_weights: Tensor                      # Type hint for registered class weights buffer

    def __init__(
        self,
        gamma: float,
        reduction: str,
        class_weights: Tensor,
    ) -> None:
        """
        Initialize Class-Balanced Focal Loss.

        Args:
            gamma:
                Focusing parameter.

            reduction:
                Reduction method.

            class_weights:
                Precomputed class weights.

        Raises:
            LossInitializationError:
                If configuration parameters are invalid.
        """

        super().__init__()                      # Initialize parent PyTorch nn.Module class

        if gamma < 0:
            raise LossInitializationError(      # Validate that gamma focus parameter is non-negative
                "Gamma must be non-negative."
            )

        supported_reductions = {                # Set of valid reduction modes
            "mean",
            "sum",
            "none",
        }

        if reduction not in supported_reductions:
            raise LossInitializationError(      # Validate reduction parameter against allowed set
                f"Unsupported reduction: "
                f"{reduction}"
            )

        self._gamma = gamma                     # Store focusing parameter
        self._reduction = reduction             # Store reduction strategy name

        self.register_buffer(                   # Register class weights as module persistent buffer
            "_class_weights",
            class_weights,
        )

    def forward(
        self,
        logits: Tensor,
        targets: Tensor,
    ) -> Tensor:
        """
        Compute Class-Balanced Focal Loss.

        Args:
            logits:
                Raw model predictions.

            targets:
                Ground-truth class indices.

        Returns:
            Computed loss.
        """

        cross_entropy_loss = F.cross_entropy(   # Calculate unreduced weighted cross entropy
            logits,
            targets,
            weight=self._class_weights,
            reduction="none",
        )

        pt = torch.exp(                         # Estimate class probabilities p_t from CE loss
            -cross_entropy_loss,
        )

        focal_loss = (                          # Apply focal weighting factor (1 - p_t)^gamma
            (1.0 - pt) ** self._gamma
        ) * cross_entropy_loss

        if self._reduction == "mean":
            return focal_loss.mean()            # Return mean loss across batch

        if self._reduction == "sum":
            return focal_loss.sum()             # Return total sum of losses across batch

        return focal_loss                       # Return raw loss tensor without reduction


def build_class_balanced_focal_loss(
    config: LossConfig,
    class_weights: Tensor | None = None,
) -> nn.Module:
    """
    Build a Class-Balanced Focal Loss instance.

    Args:
        config:
            Loss configuration.

        class_weights:
            Precomputed class weights.

    Returns:
        Configured Class-Balanced Focal Loss instance.

    Raises:
        LossInitializationError:
            If class weights are not provided.
    """

    if class_weights is None:
        raise LossInitializationError(          # Ensure class weights are explicitly provided
            "Class-Balanced Focal Loss "
            "requires class weights."
        )

    return ClassBalancedFocalLoss(              # Instantiate ClassBalancedFocalLoss module
        gamma=config.gamma,
        reduction=config.reduction,
        class_weights=class_weights,
    )