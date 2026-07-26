"""
Unit tests for the trainer state tracking module.

This module contains unit tests verifying that `TrainerState` correctly initializes with default values,
increments epoch and global step counters, updates the best validation metrics along with the best epoch index,
and accurately stores historical epoch performance outputs.
"""

from __future__ import annotations  # Enables modern type hints (Python 3.7+)

from trainer.state import TrainerState  # Dataclass tracking training run lifecycle and history
from trainer.types import EpochOutput  # Container storing aggregated epoch metrics and statistics


def test_initial_state() -> None:
    """
    Verify that TrainerState initializes with correct default attribute values.
    """

    state = TrainerState()

    assert state.current_epoch == 0
    assert state.global_step == 0
    assert state.best_metric == float("-inf")
    assert state.best_epoch == 0
    assert state.train_loss == 0.0
    assert state.validation_loss == 0.0
    assert state.history == []


def test_update_best() -> None:
    """
    Verify that update_best correctly identifies improved metrics and retains historical best values.
    """

    state = TrainerState()

    # Verify updating with an improved metric
    state.current_epoch = 1
    updated = state.update_best(0.75)

    assert updated is True
    assert state.best_metric == 0.75
    assert state.best_epoch == 1

    # Verify lower metric value is ignored
    state.current_epoch = 2
    updated = state.update_best(0.70)

    assert updated is False
    assert state.best_metric == 0.75
    assert state.best_epoch == 1


def test_next_epoch() -> None:
    """
    Verify that calling next_epoch increments the epoch counter sequentially.
    """

    state = TrainerState()

    state.next_epoch()
    assert state.current_epoch == 1

    state.next_epoch()
    assert state.current_epoch == 2


def test_increment_step() -> None:
    """
    Verify that calling increment_step increments the global optimization step counter sequentially.
    """

    state = TrainerState()

    state.increment_step()
    assert state.global_step == 1

    state.increment_step()
    assert state.global_step == 2


def test_add_epoch_result() -> None:
    """
    Verify that epoch output records are correctly appended to the history list.
    """

    state = TrainerState()

    result = EpochOutput(
        loss=0.5,
        metrics={
            "accuracy": 0.9,
        },
    )

    state.add_epoch_result(result)

    assert len(state.history) == 1
    assert state.history[0].loss == 0.5
    assert state.history[0].metrics["accuracy"] == 0.9