"""
Unit tests for trainer module type definitions and dataclass structures.

This module contains unit tests verifying the instantiation, attribute assignment, and state
retrieval of dataclasses defined in the trainer module (`StepOutput`, `EpochOutput`, `TrainingOutput`).
"""

from __future__ import annotations  # Enables modern type hints (Python 3.7+)

from trainer.types import (  # Dataclass containers for step, epoch, and overall training outputs
    EpochOutput,
    StepOutput,
    TrainingOutput,
)


def test_step_output() -> None:
    """
    Verify that StepOutput correctly stores batch loss and size statistics.
    """

    output = StepOutput(
        loss=1.25,
        batch_size=8,
    )

    assert output.loss == 1.25
    assert output.batch_size == 8


def test_epoch_output() -> None:
    """
    Verify that EpochOutput correctly stores epoch-level aggregated loss and evaluation metrics.
    """

    output = EpochOutput(
        loss=0.75,
        metrics={
            "accuracy": 0.9,
            "f1": 0.85,
        },
    )

    assert output.loss == 0.75
    assert output.metrics["accuracy"] == 0.9
    assert output.metrics["f1"] == 0.85


def test_training_output() -> None:
    """
    Verify that TrainingOutput correctly stores final training run summary statistics.
    """

    output = TrainingOutput(
        best_epoch=12,
        best_metric=0.91,
        epochs_completed=20,
    )

    assert output.best_epoch == 12
    assert output.best_metric == 0.91
    assert output.epochs_completed == 20

    print(TrainingOutput.__annotations__)
