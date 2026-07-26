"""
Training progress tracking module.

This module provides the TrainerState class, which stores the training status,
including epoch counters, step numbers, best validation scores, and loss history.
"""

from __future__ import annotations        # Enables modern type hints

from dataclasses import dataclass, field  # Utility for creating clean data container classes

from trainer.types import EpochOutput     # Container for epoch loss and calculated metrics


@dataclass(slots=True)
class TrainerState:
    """
    Stores and manages all changing variables during model training.
    """

    # Counters for training progress
    current_epoch: int = 0
    global_step: int = 0

    # Best validation results tracking
    best_metric: float = float("-inf")  # Initialized to negative infinity so any initial score is higher
    best_epoch: int = 0

    # Loss values for the current epoch
    train_loss: float = 0.0
    validation_loss: float = 0.0

    # History of all completed epoch outputs
    history: list[EpochOutput] = field(default_factory=list)

    def update_best(
        self,
        metric: float,
    ) -> bool:
        """
        Check if the new metric value exceeds the current best result.

        Returns True if a new record is established, signaling that a checkpoint should be saved.
        """

        # Compare the new validation score with the previous best score
        if metric > self.best_metric:
            self.best_metric = metric
            self.best_epoch = self.current_epoch
            return True  # A new best result was saved

        return False  # No improvement recorded

    def next_epoch(self) -> None:
        """
        Increment the current epoch counter by one.
        """

        self.current_epoch += 1

    def increment_step(self) -> None:
        """
        Increment the total step counter by one after processing a batch.
        """

        self.global_step += 1

    def add_epoch_result(
        self,
        result: EpochOutput,
    ) -> None:
        """
        Append the completed epoch summary to the history log.
        """

        self.history.append(result)