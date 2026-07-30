"""Tests for trainer utility functions."""

from __future__ import annotations

import torch
from torch import nn

from trainer.utils import (
    calculate_average_loss,
    detach_loss,
    move_to_device,
    set_eval_mode,
    set_train_mode,
)


def test_move_to_device() -> None:
    """Verify tensors are moved to the requested device."""

    tensor = torch.tensor(
        [1.0, 2.0, 3.0],
    )

    moved = move_to_device(
        tensor,
        "cpu",
    )

    assert moved.device.type == "cpu"


def test_set_train_mode() -> None:
    """Verify model is switched to training mode."""

    model = nn.Linear(
        4,
        2,
    )

    set_train_mode(model)

    assert model.training is True


def test_set_eval_mode() -> None:
    """Verify model is switched to evaluation mode."""

    model = nn.Linear(
        4,
        2,
    )

    set_eval_mode(model)

    assert model.training is False


def test_detach_loss() -> None:
    """Verify loss tensor is converted to float."""

    loss = torch.tensor(
        1.25,
        requires_grad=True,
    )

    value = detach_loss(loss)

    assert isinstance(value, float)
    assert value == 1.25


def test_calculate_average_loss() -> None:
    """Verify average loss is computed correctly."""

    assert (
        calculate_average_loss(
            6.0,
            3,
        )
        == 2.0
    )

    assert (
        calculate_average_loss(
            0.0,
            0,
        )
        == 0.0
    )
