"""Tests for the trainer module."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.optim import SGD

from metrics.accuracy import compute_accuracy
from trainer.trainer import Trainer
from trainer.types import TrainingOutput


@dataclass(slots=True)
class _Batch:
    """
    Minimal batch container used by trainer tests.
    """

    image: torch.Tensor
    label: torch.Tensor


def _build_batches() -> list[_Batch]:
    """
    Build a small synthetic batch list for testing.
    """

    return [
        _Batch(
            image=torch.tensor(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                ],
                dtype=torch.float32,
            ),
            label=torch.tensor(
                [
                    0,
                    0,
                ],
                dtype=torch.long,
            ),
        ),
    ]


def _build_trainer(tmp_path: Path) -> Trainer:
    """
    Build a trainer instance with deterministic test components.
    """

    model = nn.Linear(
        4,
        3,
    )

    with torch.no_grad():
        model.weight.zero_()
        model.bias.copy_(
            torch.tensor(
                [
                    5.0,
                    0.0,
                    0.0,
                ]
            )
        )

    optimizer = SGD(
        model.parameters(),
        lr=0.1,
    )

    criterion = nn.CrossEntropyLoss()

    batches = _build_batches()

    metrics = {
        "accuracy": compute_accuracy,
    }

    return Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=None,
        train_loader=batches,
        validation_loader=batches,
        metrics=metrics,
        device="cpu",
        checkpoint_directory=tmp_path,
    )


def test_fit_updates_state_and_returns_summary(
    tmp_path: Path,
) -> None:
    """
    Verify fit() runs end-to-end and updates trainer state.
    """

    trainer = _build_trainer(
        tmp_path,
    )

    output = trainer.fit(
        epochs=2,
        monitor="accuracy",
    )

    assert isinstance(
        output,
        TrainingOutput,
    )

    assert output.epochs_completed == 2
    assert output.best_epoch == 1
    assert output.best_metric == pytest.approx(
        1.0,
    )

    assert trainer.state.current_epoch == 2
    assert trainer.state.global_step == 2
    assert trainer.state.best_epoch == 1
    assert trainer.state.best_metric == pytest.approx(
        1.0,
    )

    assert len(trainer.state.history) == 2
    assert trainer.state.history[0].metrics["accuracy"] == pytest.approx(
        1.0,
    )


def test_checkpoint_round_trip(
    tmp_path: Path,
) -> None:
    """
    Verify checkpoint save/load restores model parameters and state.
    """

    trainer = _build_trainer(
        tmp_path,
    )

    trainer.fit(
        epochs=1,
        monitor="accuracy",
    )

    checkpoint_path = tmp_path / "trainer_checkpoint.pt"

    original_state = {
        name: tensor.detach().clone()
        for name, tensor in trainer._model.state_dict().items()
    }

    trainer.save_checkpoint(
        checkpoint_path,
    )

    assert checkpoint_path.exists()

    with torch.no_grad():
        for parameter in trainer._model.parameters():
            parameter.add_(
                10.0,
            )

    trainer.load_checkpoint(
        checkpoint_path,
    )

    restored_state = trainer._model.state_dict()

    for name, tensor in original_state.items():
        assert torch.equal(
            restored_state[name],
            tensor,
        )

    assert trainer.state.best_epoch == 1
    assert trainer.state.best_metric == pytest.approx(
        1.0,
    )