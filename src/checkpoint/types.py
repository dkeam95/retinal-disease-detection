"""
Type definitions for the checkpoint module.

This module defines immutable data structures (dataclasses) used to convey
checkpoint metadata and complete serialized training state packages.
"""

from __future__ import (
    annotations,  # Enables modern union types (|) and postponed annotation evaluation
)

from dataclasses import dataclass  # Decorator for creating immutable data containers
from pathlib import Path  # Object-oriented filesystem path handler
from typing import Any  # Generic type hint for state dictionary values

from trainer.state import (
    TrainerState,  # Class tracking training progress and epoch metrics
)


@dataclass(frozen=True, slots=True)
class CheckpointMetadata:
    """
    Metadata describing a checkpoint file stored on disk.

    Attributes
    ----------
    path : Path
        Absolute or relative file path to the saved checkpoint.
    epoch : int
        Epoch index at which the checkpoint was captured.
    metric : float
        Validation metric score achieved at the time of saving.
    """

    path: Path
    epoch: int
    metric: float


@dataclass(frozen=True, slots=True)
class CheckpointData:
    """
    Complete in-memory container holding all serialized state components of a checkpoint.

    Attributes
    ----------
    model_state : dict[str, Any]
        Serialized model parameter state dictionary.
    optimizer_state : dict[str, Any]
        Serialized optimizer state dictionary.
    scheduler_state : dict[str, Any] | None
        Serialized learning rate scheduler state dictionary, or None if unused.
    trainer_state : TrainerState
        Snapshot of the trainer state including progress metrics and history.
    metadata : CheckpointMetadata
        Associated metadata containing path, epoch, and metric details.
    """

    model_state: dict[str, Any]
    optimizer_state: dict[str, Any]
    scheduler_state: dict[str, Any] | None
    trainer_state: TrainerState
    metadata: CheckpointMetadata
