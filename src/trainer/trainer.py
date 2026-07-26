"""
Main trainer implementation module.

This module contains the Trainer class, which coordinates model training,
validation evaluation, learning rate scheduling, and saving or loading checkpoints.
"""

from __future__ import annotations    # Enables modern type hints

from collections.abc import Callable  # Type hint for metric evaluation functions
from pathlib import Path              # File system path utility

import torch                            # PyTorch core library
from torch import nn                    # Base classes for neural network modules
from torch.optim import Optimizer       # Base class for optimization algorithms
from torch.optim.lr_scheduler import LRScheduler  # Base class for learning rate schedulers
from torch.utils.data import DataLoader           # PyTorch data loader class

from trainer.exceptions import CheckpointError      # Custom exception for checkpoint saving/loading failures
from trainer.state import TrainerState              # Class tracking training progress and metric history
from trainer.step import (                          # Functions for processing single training and validation steps
    train_step,
    validation_step,
)
from trainer.types import EpochOutput, TrainingOutput  # Containers for aggregated training statistics
from trainer.utils import (                            # Utility functions for device transfer and model mode switching
    calculate_average_loss,
    move_to_device,
    set_eval_mode,
    set_train_mode,
)


class Trainer:
    """
    Main controller for training and evaluating a PyTorch model.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        criterion: nn.Module,
        scheduler: LRScheduler | None,
        train_loader: DataLoader,
        validation_loader: DataLoader,
        metrics: dict[str, Callable],
        device: str,
        checkpoint_directory: Path | None = None,
    ) -> None:
        """
        Initialize the trainer with all required components.

        Parameters
        ----------
        model : nn.Module
            Neural network model instance.
        optimizer : Optimizer
            Optimization algorithm instance.
        criterion : nn.Module
            Loss function module.
        scheduler : LRScheduler | None
            Optional learning rate scheduler.
        train_loader : DataLoader
            Data loader for the training dataset.
        validation_loader : DataLoader
            Data loader for the validation dataset.
        metrics : dict[str, Callable]
            Dictionary mapping metric names to evaluation functions.
        device : str
            Target computing device (e.g., 'cuda' or 'cpu').
        checkpoint_directory : Path | None, optional
            Directory path for saving checkpoints, by default None.
        """

        # Model and optimization components
        self._model = model
        self._optimizer = optimizer
        self._criterion = criterion
        self._scheduler = scheduler

        # Data loaders
        self._train_loader = train_loader
        self._validation_loader = validation_loader

        # Metrics and execution target
        self._metrics = metrics
        self._device = torch.device(device)
        self._checkpoint_directory = checkpoint_directory

        # State tracker for epochs, steps, and metrics
        self._state = TrainerState()

        # Move model parameters to the target execution device
        self._model.to(self._device)

    def _train_epoch(self) -> EpochOutput:
        """
        Run one full training epoch over the training dataset.

        Returns
        -------
        EpochOutput
            Average training loss statistics for the epoch.
        """

        # Set model to training mode (enables Dropout and updates BatchNorm)
        set_train_mode(self._model)

        total_loss = 0.0
        total_batches = 0

        # Iterate over all training batches
        for batch in self._train_loader:
            # Move inputs and labels to the target device
            images = move_to_device(
                batch.image,
                self._device,
            )

            targets = move_to_device(
                batch.label,
                self._device,
            )

            # Perform a single forward/backward pass and step update
            step_output = train_step(
                model=self._model,
                optimizer=self._optimizer,
                criterion=self._criterion,
                images=images,
                targets=targets,
            )

            total_loss += step_output.loss
            total_batches += 1
            self._state.increment_step()

        # Compute average loss across all batches
        average_loss = calculate_average_loss(
            total_loss,
            total_batches,
        )

        return EpochOutput(
            loss=average_loss,
            metrics={},
        )

    def _validate_epoch(self) -> EpochOutput:
        """
        Run one full evaluation epoch over the validation dataset.

        Returns
        -------
        EpochOutput
            Average validation loss and calculated evaluation metrics.
        """

        # Set model to evaluation mode (disables Dropout and freezes BatchNorm)
        set_eval_mode(self._model)

        total_loss = 0.0
        total_batches = 0

        all_logits: list[torch.Tensor] = []
        all_targets: list[torch.Tensor] = []

        # Iterate over all validation batches
        for batch in self._validation_loader:
            # Move inputs and labels to the target device
            images = move_to_device(
                batch.image,
                self._device,
            )

            targets = move_to_device(
                batch.label,
                self._device,
            )

            # Perform a single forward pass under no_grad context
            logits, step_output = validation_step(
                model=self._model,
                criterion=self._criterion,
                images=images,
                targets=targets,
            )

            total_loss += step_output.loss
            total_batches += 1

            # Move predictions to CPU memory to prevent GPU memory overflow
            all_logits.append(logits.detach().cpu())
            all_targets.append(targets.detach().cpu())

        # Compute average validation loss
        average_loss = calculate_average_loss(
            total_loss,
            total_batches,
        )

        # Concatenate predictions and calculate evaluation metrics
        if all_logits:
            logits_tensor = torch.cat(all_logits, dim=0)
            targets_tensor = torch.cat(all_targets, dim=0)

            metrics = {
                metric_name: metric_fn(logits_tensor, targets_tensor)
                for metric_name, metric_fn in self._metrics.items()
            }
        else:
            metrics = {}

        return EpochOutput(
            loss=average_loss,
            metrics=metrics,
        )

    def fit(
        self,
        epochs: int,
        monitor: str = "accuracy",
    ) -> TrainingOutput:
        """
        Run the complete training loop for a specified number of epochs.

        Parameters
        ----------
        epochs : int
            Total number of epochs to train.
        monitor : str, optional
            Metric name used to track the best model performance, by default "accuracy".

        Returns
        -------
        TrainingOutput
            Summary report with total completed epochs and best metric score.
        """

        for _ in range(epochs):
            # Advance epoch counter
            self._state.next_epoch()

            # Execute training and validation phases
            train_output = self._train_epoch()
            validation_output = self._validate_epoch()

            # Save epoch loss values to state
            self._state.train_loss = train_output.loss
            self._state.validation_loss = validation_output.loss

            # Store validation output in history
            self._state.add_epoch_result(validation_output)

            # Check if current validation score sets a new record
            monitored_metric = validation_output.metrics.get(monitor)

            if monitored_metric is not None:
                self._state.update_best(
                    monitored_metric,
                )

            # Update learning rate if scheduler is provided
            if self._scheduler is not None:
                self._scheduler.step()

        return TrainingOutput(
            best_epoch=self._state.best_epoch,
            best_metric=self._state.best_metric,
            epochs_completed=self._state.current_epoch,
        )

    def save_checkpoint(
        self,
        path: Path,
    ) -> None:
        """
        Save the current model, optimizer, scheduler, and trainer states to a file.

        Parameters
        ----------
        path : Path
            File path where the checkpoint will be saved.

        Raises
        ------
        CheckpointError
            If saving the checkpoint to disk fails.
        """

        # Prepare state dictionary containing all components
        checkpoint = {
            "model_state_dict": self._model.state_dict(),
            "optimizer_state_dict": self._optimizer.state_dict(),
            "scheduler_state_dict": (
                self._scheduler.state_dict()
                if self._scheduler is not None
                else None
            ),
            "trainer_state": self._state,
        }

        try:
            torch.save(checkpoint, path)

        except (OSError, RuntimeError) as error:
            raise CheckpointError(
                f"Failed to save checkpoint: {path}"
            ) from error

    def load_checkpoint(
        self,
        path: Path,
    ) -> None:
        """
        Restore model, optimizer, scheduler, and trainer states from a checkpoint file.

        Parameters
        ----------
        path : Path
            File path of the checkpoint to load.

        Raises
        ------
        CheckpointError
            If reading the file or restoring states fails.
        """

        try:
            # Load checkpoint dictionary to the target device
            checkpoint = torch.load(path, map_location=self._device, weights_only=False)

        except (OSError, RuntimeError) as error:
            raise CheckpointError(
                f"Failed to load checkpoint: {path}"
            ) from error

        try:
            # Restore states for model and optimizer
            self._model.load_state_dict(
                checkpoint["model_state_dict"]
            )
            self._optimizer.load_state_dict(
                checkpoint["optimizer_state_dict"]
            )

            # Restore scheduler state if applicable
            if (
                self._scheduler is not None
                and checkpoint["scheduler_state_dict"] is not None
            ):
                self._scheduler.load_state_dict(
                    checkpoint["scheduler_state_dict"],
                )

            # Restore trainer state tracking
            self._state = checkpoint["trainer_state"]

        except (KeyError, TypeError, RuntimeError) as error:
            raise CheckpointError(
                f"Invalid checkpoint format: {path}"
            ) from error

    @property
    def state(self) -> TrainerState:
        """
        Get the current trainer state instance.

        Returns
        -------
        TrainerState
            Object tracking current epoch, step, metrics, and loss history.
        """

        return self._state