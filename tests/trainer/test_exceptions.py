"""
Unit tests for trainer module custom exceptions.

This module contains unit tests verifying the raise behaviors and sub-classing hierarchy
of custom exceptions defined in the trainer module.
"""

from __future__ import annotations  # Enables modern type hints (Python 3.7+)

import pytest  # PyTest framework for test execution and exception assertions

from trainer.exceptions import (  # Custom trainer exception hierarchy
    CheckpointError,
    EarlyStoppingError,
    InvalidTrainerStateError,
    TrainerError,
    TrainingStepError,
    ValidationStepError,
)


def test_invalid_trainer_state_error() -> None:
    """
    Verify that InvalidTrainerStateError can be instantiated and raised.
    """

    with pytest.raises(
        InvalidTrainerStateError,
    ):
        raise InvalidTrainerStateError(
            "Invalid trainer state.",
        )


def test_training_step_error() -> None:
    """
    Verify that TrainingStepError can be instantiated and raised.
    """

    with pytest.raises(
        TrainingStepError,
    ):
        raise TrainingStepError(
            "Training step failed.",
        )


def test_validation_step_error() -> None:
    """
    Verify that ValidationStepError can be instantiated and raised.
    """

    with pytest.raises(
        ValidationStepError,
    ):
        raise ValidationStepError(
            "Validation step failed.",
        )


def test_checkpoint_error() -> None:
    """
    Verify that CheckpointError can be instantiated and raised.
    """

    with pytest.raises(
        CheckpointError,
    ):
        raise CheckpointError(
            "Checkpoint operation failed.",
        )


def test_early_stopping_error() -> None:
    """
    Verify that EarlyStoppingError can be instantiated and raised.
    """

    with pytest.raises(
        EarlyStoppingError,
    ):
        raise EarlyStoppingError(
            "Early stopping failed.",
        )


def test_trainer_error_inheritance() -> None:
    """
    Verify that all specific trainer exception classes inherit from the base TrainerError.
    """

    assert issubclass(
        InvalidTrainerStateError,
        TrainerError,
    )

    assert issubclass(
        TrainingStepError,
        TrainerError,
    )

    assert issubclass(
        ValidationStepError,
        TrainerError,
    )

    assert issubclass(
        CheckpointError,
        TrainerError,
    )

    assert issubclass(
        EarlyStoppingError,
        TrainerError,
    )
