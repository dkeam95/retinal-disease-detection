"""
Type definitions and data containers module.

This module defines lightweight dataclass structures used to store step statistics,
aggregated epoch metrics, and final training outputs across the pipeline.
"""

from __future__ import annotations  # Enables modern type hints

from dataclasses import dataclass  # Decorator for creating data container classes


@dataclass(slots=True)
class StepOutput:
    """
    Stores loss and batch size for a single training or validation step.

    Attributes
    ----------
    loss : float
        Scalar loss value calculated for the current batch.
    batch_size : int
        Number of samples processed in the batch.
    """

    loss: float
    batch_size: int


@dataclass(slots=True)
class EpochOutput:
    """
    Stores average loss and evaluation metrics for a completed epoch.

    Attributes
    ----------
    loss : float
        Average loss across all batches in the epoch.
    metrics : dict[str, float]
        Dictionary mapping metric names to their calculated values.
    """

    loss: float
    metrics: dict[str, float]


@dataclass(slots=True)
class TrainingOutput:
    """
    Final result returned after training finishes.
    """

    best_epoch: int
    best_metric: float
    epochs_completed: int
