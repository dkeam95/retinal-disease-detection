"""
Unit tests for the checkpoint manager.

This module verifies checkpoint saving, loading, and complete
training state restoration.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn
from torch.optim import SGD

from checkpoint.manager import CheckpointManager
from trainer.state import TrainerState


def _build_model() -> nn.Module:
    """
    Build a simple test model.
    """

    return nn.Linear(
        4,
        3,
    )


def _build_optimizer(
    model: nn.Module,
) -> SGD:
    """
    Build optimizer.
    """

    return SGD(
        model.parameters(),
        lr=0.01,
    )


def _build_state() -> TrainerState:
    """
    Build trainer state.
    """

    state = TrainerState()

    state.current_epoch = 5
    state.global_step = 100
    state.best_epoch = 4
    state.best_metric = 0.91

    return state


def test_save_checkpoint(
    tmp_path: Path,
) -> None:
    """
    Verify checkpoint is successfully saved.
    """

    manager = CheckpointManager(
        checkpoint_directory=tmp_path,
    )

    model = _build_model()
    optimizer = _build_optimizer(
        model,
    )
    state = _build_state()

    checkpoint_path = manager.save(
        model=model,
        optimizer=optimizer,
        scheduler=None,
        trainer_state=state,
    )

    assert checkpoint_path.exists()
    assert checkpoint_path.is_file()


def test_load_checkpoint(
    tmp_path: Path,
) -> None:
    """
    Verify checkpoint restores trainer state.
    """

    manager = CheckpointManager(
        checkpoint_directory=tmp_path,
    )

    model = _build_model()
    optimizer = _build_optimizer(
        model,
    )
    state = _build_state()

    checkpoint_path = manager.save(
        model=model,
        optimizer=optimizer,
        scheduler=None,
        trainer_state=state,
    )

    restored_state = manager.load(
        checkpoint_path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        scheduler=None,
    )

    assert restored_state.current_epoch == 5
    assert restored_state.global_step == 100
    assert restored_state.best_epoch == 4
    assert restored_state.best_metric == pytest.approx(
        0.91,
    )


def test_load_restores_model_weights(
    tmp_path: Path,
) -> None:
    """
    Verify model parameters are restored.
    """

    manager = CheckpointManager(
        checkpoint_directory=tmp_path,
    )

    model = _build_model()
    optimizer = _build_optimizer(
        model,
    )

    state = _build_state()

    original_state = {key: value.clone() for key, value in model.state_dict().items()}

    checkpoint_path = manager.save(
        model=model,
        optimizer=optimizer,
        scheduler=None,
        trainer_state=state,
    )

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(10.0)

    manager.load(
        checkpoint_path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        scheduler=None,
    )

    restored_state = model.state_dict()

    for key in original_state:
        assert torch.equal(
            original_state[key],
            restored_state[key],
        )


def test_load_missing_checkpoint(
    tmp_path: Path,
) -> None:
    """
    Verify loading missing checkpoint raises an exception.
    """

    manager = CheckpointManager(
        checkpoint_directory=tmp_path,
    )

    model = _build_model()
    optimizer = _build_optimizer(
        model,
    )

    with pytest.raises(Exception):
        manager.load(
            checkpoint_path=tmp_path / "missing.pt",
            model=model,
            optimizer=optimizer,
            scheduler=None,
        )
