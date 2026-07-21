"""Unit tests for Class-Balanced Focal Loss."""

from __future__ import annotations

import torch

from common.config.types import LossConfig
from losses.class_balanced_focal import (
    build_class_balanced_focal_loss,
)


def test_weighted_loss_is_computed() -> None:
    """Verify that class weights affect the loss."""

    logits = torch.tensor(
        [
            [2.0, 0.1],
            [2.0, 0.1],
        ],
        dtype=torch.float32,
    )

    targets = torch.tensor(
        [0, 1],
        dtype=torch.long,
    )

    class_weights = torch.tensor(
        [1.0, 5.0],
        dtype=torch.float32,
    )

    config = LossConfig(
        name="class_balanced_focal",
        gamma=2.0,
        reduction="none",
    )

    criterion = build_class_balanced_focal_loss(
        config,
        class_weights,
    )

    loss = criterion(logits, targets)
    print(f"\nLoss = {loss}")
    assert loss[1] > loss[0]


def test_backward_loss() -> None:
    """Verify that gradients are computed correctly."""

    logits = torch.randn(
        4,
        5,
        dtype=torch.float32,
        requires_grad=True
    )

    targets = torch.tensor(
        [0, 1, 2, 3],
        dtype=torch.long
    )

    class_weights = torch.tensor(
        [1.0, 2.0, 3.0, 4.0, 5.0],
        dtype=torch.float32
    )

    config = LossConfig(
        name="class_balanced_focal",
        gamma=2.0,
        reduction="mean"
    )

    criterion = build_class_balanced_focal_loss(
        config,
        class_weights
    )

    loss = criterion(
        logits,
        targets
    )

    print(f"\nLoss = {loss.item()}")

    loss.backward()

    print(f"\nGradient shape = {logits.grad.shape}")

    assert logits.grad is not None
    assert logits.grad.shape == logits.shape
    print("\nGradient computed successfully")


def test_reduction_mean() -> None:
    """Verify reduction='mean' returns a scalar."""

    logits = torch.randn(
        8,
        5,
        dtype=torch.float32,
    )

    targets = torch.randint(
        0,
        5,
        (8,),
        dtype=torch.long,
    )

    class_weights = torch.ones(
        5,
        dtype=torch.float32,
    )

    config = LossConfig(
        name="class_balanced_focal",
        gamma=2.0,
        reduction="mean",
    )

    criterion = build_class_balanced_focal_loss(
        config,
        class_weights,
    )

    loss = criterion(
        logits,
        targets,
    )

    print(f"\nLoss shape = {loss.shape}")

    assert loss.ndim == 0


def test_reduction_none() -> None:
    """Verify reduction='none' returns per-sample losses."""

    logits = torch.randn(
        8,
        5,
        dtype=torch.float32,
    )

    targets = torch.randint(
        0,
        5,
        (8,),
        dtype=torch.long,
    )

    class_weights = torch.ones(
        5,
        dtype=torch.float32,
    )

    config = LossConfig(
        name="class_balanced_focal",
        gamma=2.0,
        reduction="none",
    )

    criterion = build_class_balanced_focal_loss(
        config,
        class_weights,
    )

    loss = criterion(
        logits,
        targets,
    )

    print(f"\nLoss shape = {loss.shape}")

    assert loss.ndim == 1
    assert loss.shape[0] == targets.shape[0]


import pytest


def test_invalid_reduction() -> None:
    """Verify invalid reduction raises ValueError."""

    logits = torch.randn(
        4,
        5,
        dtype=torch.float32,
    )

    targets = torch.randint(
        0,
        5,
        (4,),
        dtype=torch.long,
    )

    class_weights = torch.ones(
        5,
        dtype=torch.float32,
    )

    config = LossConfig(
        name="class_balanced_focal",
        gamma=2.0,
        reduction="invalid",
    )

    criterion = build_class_balanced_focal_loss(
        config,
        class_weights,
    )

    with pytest.raises(ValueError):
        criterion(
            logits,
            targets,
        )

    print("\nValueError successfully raised.")