"""
Focal Loss implementation.
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


class FocalLoss(nn.Module):
    """
    Implementation of Focal Loss.

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
        self._alpha = alpha                     # Store optional alpha balancing factor
        self._reduction = reduction             # Store reduction strategy name

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

        cross_entropy_loss = F.cross_entropy(   # Calculate unreduced standard cross entropy
            logits,
            targets,
            reduction="none",
        )

        pt = torch.exp(                         # Estimate class probabilities p_t from CE loss
            -cross_entropy_loss,
        )

        focal_loss = (                          # Apply focal weighting factor (1 - p_t)^gamma
            (1.0 - pt) ** self._gamma
        ) * cross_entropy_loss

        if self._alpha is not None:
            focal_loss = (                      # Scale loss by alpha balancing factor if defined
                self._alpha
                * focal_loss
            )

        if self._reduction == "mean":
            return focal_loss.mean()            # Return mean loss across batch

        if self._reduction == "sum":
            return focal_loss.sum()             # Return total sum of losses across batch

        return focal_loss                       # Return raw loss tensor without reduction


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

    return FocalLoss(                           # Instantiate and return FocalLoss module
        gamma=config.gamma,
        alpha=config.alpha,
        reduction=config.reduction,
    )