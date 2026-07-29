"""
Trainer module public API.

This module exposes the primary training loop utilities, custom error types, state managers,
and execution step functions used across the training pipeline.
"""

from __future__ import annotations  # Enables modern type hints (Python 3.7+)

from trainer.exceptions import (  # Custom exception hierarchy for trainer-related errors
    CheckpointError,
    EarlyStoppingError,
    InvalidTrainerStateError,
    TrainerError,
    TrainingStepError,
    ValidationStepError,
)
from trainer.state import (
    TrainerState,  # Dataclass tracking training run lifecycle and history
)
from trainer.step import (  # Individual training and validation step functions
    train_step,
    validation_step,
)
from trainer.trainer import Trainer  # Main high-level trainer orchestrator class
from trainer.types import (  # Typed containers for step, epoch, and overall training outputs
    EpochOutput,
    StepOutput,
    TrainingOutput,
)

__all__ = [
    # Core Orchestrator
    "Trainer",
    # State & Context
    "TrainerState",
    # Step Execution Functions
    "train_step",
    "validation_step",
    # Output Data Containers
    "EpochOutput",
    "StepOutput",
    "TrainingOutput",
    # Error Hierarchy
    "CheckpointError",
    "EarlyStoppingError",
    "InvalidTrainerStateError",
    "TrainerError",
    "TrainingStepError",
    "ValidationStepError",
]
