"""
Unit tests for the Focal Loss implementation.

This module contains unit tests verifying proper behavior of Focal Loss, including
mathematical equivalence to CrossEntropyLoss when gamma=0, reduction modes ('mean',
'sum', 'none'), gradient flow during backpropagation, non-negativity guarantees,
and near-zero loss on confident correct predictions.
"""

from __future__ import annotations  # Enables modern type hints (Python 3.7+)

import torch  # Core PyTorch library

from common.config.types import LossConfig  # Dataclass/type defining loss parameters
from losses.focal import (
    build_focal_loss,  # Factory function that instantiates Focal Loss
)


def test_gamma_zero_matches_cross_entropy() -> None:
    """
    Verify that Focal Loss with gamma=0 produces results identical to standard CrossEntropyLoss.
    """

    # Mock model predictions (2 samples, 3 classes)
    logits = torch.tensor(
        [
            [2.0, 0.5, -1.0],
            [0.1, 1.5, 0.3],
        ],
        dtype=torch.float32,
    )

    # True class labels (sample 0 -> class 0, sample 1 -> class 1)
    targets = torch.tensor(
        [0, 1],
        dtype=torch.long,
    )

    # Configuration for Focal Loss with gamma=0.0
    config = LossConfig(
        name="focal",
        alpha=None,
        gamma=0.0,
        reduction="mean",
    )

    # Instantiate custom Focal Loss
    focal_loss = build_focal_loss(config)

    # Instantiate PyTorch's standard CrossEntropyLoss
    cross_entropy = torch.nn.CrossEntropyLoss()

    # Calculate loss values
    focal_value = focal_loss(logits, targets)
    cross_entropy_value = cross_entropy(logits, targets)

    # Assert values are practically equal within float32 tolerance (1e-6)
    assert torch.allclose(
        focal_value,
        cross_entropy_value,
        atol=1e-6,
    )


def test_reduction_mean() -> None:
    """
    Verify that reduction='mean' averages loss across the batch to return a scalar.
    """

    # Generate random logits batch (4 samples, 5 classes)
    logits = torch.randn(
        4,
        5,
        dtype=torch.float32,
    )
    targets = torch.tensor(
        [0, 1, 2, 3],
        dtype=torch.long,
    )

    config = LossConfig(
        name="focal",
        gamma=2.0,
        alpha=None,
        reduction="mean",
    )

    criterion = build_focal_loss(config)
    loss = criterion(logits, targets)

    # ndim == 0 verifies the result is a scalar (0D tensor)
    assert loss.ndim == 0


def test_reduction_sum() -> None:
    """
    Verify that reduction='sum' accumulates losses across the batch to return a scalar.
    """

    # Generate random logits batch (4 samples, 5 classes)
    logits = torch.randn(
        4,
        5,
        dtype=torch.float32,
    )
    targets = torch.tensor(
        [0, 1, 2, 3],
        dtype=torch.long,
    )

    config = LossConfig(
        name="focal",
        alpha=None,
        gamma=2.0,
        reduction="sum",
    )

    criterion = build_focal_loss(config)
    loss = criterion(logits, targets)

    # Sum reduction should also yield a scalar
    assert loss.ndim == 0


def test_reduction_none() -> None:
    """
    Verify that reduction='none' returns unreduced per-sample losses.
    """

    # Generate random logits batch (4 samples, 5 classes)
    logits = torch.randn(
        4,
        5,
        dtype=torch.float32,
    )
    targets = torch.tensor(
        [0, 1, 2, 3],
        dtype=torch.long,
    )

    config = LossConfig(
        name="focal",
        alpha=None,
        gamma=2.0,
        reduction="none",
    )

    criterion = build_focal_loss(config)
    loss = criterion(logits, targets)

    # Expect a 1D tensor of shape (4,) containing loss for each sample
    assert loss.ndim == 1
    assert loss.shape == (4,)


def test_backward_pass() -> None:
    """
    Verify that gradients are correctly calculated and populated during backpropagation.
    """

    # Enable autograd tracking for backpropagation
    logits = torch.randn(
        4,
        5,
        dtype=torch.float32,
        requires_grad=True,
    )

    targets = torch.tensor(
        [0, 1, 2, 3],
        dtype=torch.long,
    )

    config = LossConfig(
        name="focal",
        alpha=None,
        gamma=2.0,
        reduction="mean",
    )

    criterion = build_focal_loss(config)
    loss = criterion(logits, targets)

    # Perform backpropagation
    loss.backward()

    # Ensure gradients were computed and match the shape of the input logits
    assert logits.grad is not None
    assert logits.grad.shape == logits.shape


def test_is_non_negative() -> None:
    """
    Verify that computed Focal Loss values are non-negative.
    """

    # Generate random logits batch (4 samples, 5 classes)
    logits = torch.randn(
        4,
        5,
        dtype=torch.float32,
    )
    targets = torch.tensor(
        [0, 1, 2, 3],
        dtype=torch.long,
    )

    config = LossConfig(
        name="focal",
        alpha=None,
        gamma=2.0,
        reduction="mean",
    )

    criterion = build_focal_loss(config)
    loss = criterion(logits, targets)

    # Loss value must never be negative
    assert loss.item() >= 0.0


def test_perfect_prediction() -> None:
    """
    Verify that highly confident, correct predictions produce near-zero loss.
    """

    # High logit values for target classes (15.0 vs -5.0)
    logits = torch.tensor(
        [
            [15.0, -5.0, -5.0, -5.0, -5.0],
            [-5.0, 15.0, -5.0, -5.0, -5.0],
        ],
        dtype=torch.float32,
    )

    targets = torch.tensor(
        [0, 1],
        dtype=torch.long,
    )

    config = LossConfig(
        name="focal",
        gamma=2.0,
        alpha=None,
        reduction="mean",
    )

    criterion = build_focal_loss(config)
    loss = criterion(logits, targets)

    # Near-perfect predictions should result in almost 0 loss (< 1e-6)
    assert loss.item() < 1e-6
