"""
Custom exceptions for the trainer module.

This module defines the custom exception hierarchy used throughout the training loop,
ensuring consistent error handling during model training steps, validation evaluation,
checkpoint persistence, state transitions, and early stopping checks.
"""

from __future__ import annotations  # Enables modern type hints (Python 3.7+)


class TrainerError(Exception):
    """Base exception for all trainer-related errors."""


class InvalidTrainerStateError(TrainerError):
    """Raised when an illegal trainer state transition or invalid state configuration occurs."""


class TrainingStepError(TrainerError):
    """Raised when an unexpected error occurs during a forward or backward training step."""


class ValidationStepError(TrainerError):
    """Raised when an error occurs during evaluation on the validation dataset split."""


class CheckpointError(TrainerError):
    """Raised when saving or restoring model checkpoint artifacts fails."""


class EarlyStoppingError(TrainerError):
    """Raised when an anomaly occurs during early stopping criteria evaluation."""
