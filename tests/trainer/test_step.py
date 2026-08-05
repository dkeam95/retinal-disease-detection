"""Tests for training and validation steps."""

from __future__ import annotations

import torch
from torch import nn
from torch.optim import SGD

from trainer.step import (
    train_step,
    validation_step,
)
from trainer.types import StepOutput


def _build_model() -> nn.Module:
    """
    Build a simple test model.
    """

    return nn.Linear(
        4,
        3,
    )


def _build_batch() -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build a small synthetic batch.
    """

    images = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )

    targets = torch.tensor(
        [
            0,
            2,
        ],
        dtype=torch.long,
    )

    return images, targets


def test_train_step_updates_parameters() -> None:
    """Verify train_step performs optimization and returns step stats."""

    model = _build_model()
    optimizer = SGD(
        model.parameters(),
        lr=0.1,
    )
    criterion = nn.CrossEntropyLoss()

    images, targets = _build_batch()

    before = [
        parameter.detach().clone()
        for parameter in model.parameters()
    ]

    output = train_step(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        images=images,
        targets=targets,
    )

    assert isinstance(output, StepOutput)
    assert output.batch_size == 2
    assert output.loss >= 0.0

    after = [
        parameter.detach().clone()
        for parameter in model.parameters()
    ]

    assert any(
        not torch.equal(before_tensor, after_tensor)
        for before_tensor, after_tensor in zip(before, after, strict=True)
    )


def test_validation_step_returns_logits_and_stats() -> None:
    """Verify validation_step returns logits and step statistics."""

    model = _build_model()
    criterion = nn.CrossEntropyLoss()

    images, targets = _build_batch()

    before = [
        parameter.detach().clone()
        for parameter in model.parameters()
    ]

    logits, output = validation_step(
        model=model,
        criterion=criterion,
        images=images,
        targets=targets,
    )

    assert isinstance(output, StepOutput)
    assert output.batch_size == 2
    assert output.loss >= 0.0

    assert logits.shape == (
        2,
        3,
    )

    after = [
        parameter.detach().clone()
        for parameter in model.parameters()
    ]

    assert all(
        torch.equal(before_tensor, after_tensor)
        for before_tensor, after_tensor in zip(before, after, strict=True)
    )
