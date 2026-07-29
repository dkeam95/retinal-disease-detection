"""
Main trainer implementation module.

This module contains the Trainer class, which coordinates model training,
validation evaluation, learning rate scheduling, mixed precision training,
early stopping, and saving or loading checkpoints.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import torch
from torch import nn
from torch.amp import GradScaler, autocast
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from checkpoint.manager import CheckpointManager
from trainer.exceptions import CheckpointError
from trainer.state import TrainerState
from trainer.step import train_step, validation_step
from trainer.types import EpochOutput, TrainingOutput
from trainer.utils import (
    calculate_average_loss,
    move_to_device,
    set_eval_mode,
    set_train_mode,
)


class Trainer:
    """
    Main controller for training, evaluating, and checkpointing PyTorch models.
    """

    def __init__(
        self,
        *,
        model: nn.Module,
        optimizer: Optimizer,
        criterion: nn.Module,
        scheduler: LRScheduler | None,
        train_loader: DataLoader,
        validation_loader: DataLoader,
        metrics: dict[str, Callable],
        checkpoint_manager: CheckpointManager | None = None,
        checkpoint_directory: Path | None = None,
        device: torch.device | str = "cpu",
        mixed_precision: bool = True,
        gradient_accumulation_steps: int = 1,
        validation_frequency: int = 1,
        early_stopping_patience: int = 10,
        early_stopping_min_delta: float = 0.0,
        log_every_n_steps: int = 20,
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
            Data loader for training dataset.
        validation_loader : DataLoader
            Data loader for validation dataset.
        metrics : dict[str, Callable]
            Dictionary mapping metric names to evaluation functions.
        checkpoint_manager : CheckpointManager | None, optional
            Checkpoint manager for saving/loading model state.
        checkpoint_directory : Path | None, optional
            Checkpoint directory path if checkpoint_manager is not provided.
        device : torch.device | str, default="cpu"
            Target computing device (e.g., 'cuda' or 'cpu').
        mixed_precision : bool, default=True
            Enable Automatic Mixed Precision (AMP) on CUDA devices.
        gradient_accumulation_steps : int, default=1
            Number of steps over which gradients are accumulated.
        validation_frequency : int, default=1
            Run validation every N epochs.
        early_stopping_patience : int, default=10
            Number of epochs with no improvement before training stops.
        early_stopping_min_delta : float, default=0.0
            Minimum change in monitored metric to qualify as an improvement.
        log_every_n_steps : int, default=20
            Frequency of logging training progress.
        """

        self._device = torch.device(device) if isinstance(device, str) else device

        # Core components
        if self._device.type == "cuda":
            self._model = model.to(self._device, memory_format=torch.channels_last)
            try:
                print("[Trainer] Attempting model compilation with PyTorch 2.x torch.compile()...", flush=True)
                import time
                t_comp_start = time.perf_counter()
                compiled_model = torch.compile(self._model, mode="default")
                t_comp_end = time.perf_counter()
                print(f"[Trainer] torch.compile() setup finished in {(t_comp_end - t_comp_start)*1000:.2f} ms", flush=True)
                self._model = compiled_model
            except Exception as error:
                import traceback
                print(f"[Trainer WARNING] torch.compile failed on Windows/PyTorch Inductor: {error}", flush=True)
                print(f"[Trainer WARNING] Stack trace:\n{traceback.format_exc()}", flush=True)
                print("[Trainer INFO] Continuing with fast channels_last CUDA model without JIT Inductor compilation.", flush=True)
        else:
            self._model = model.to(self._device)
        self._optimizer = optimizer
        self._criterion = criterion.to(self._device) if isinstance(criterion, torch.nn.Module) else criterion
        self._scheduler = scheduler

        # Data loaders
        self._train_loader = train_loader
        self._validation_loader = validation_loader

        # Evaluation & Checkpointing
        self._metrics = metrics
        if checkpoint_manager is not None:
            self._checkpoint_manager = checkpoint_manager
        else:
            chk_dir = checkpoint_directory if checkpoint_directory is not None else Path("checkpoints")
            self._checkpoint_manager = CheckpointManager(checkpoint_directory=chk_dir)

        # Hyperparameters
        self._mixed_precision = mixed_precision and self._device.type == "cuda"
        self._gradient_accumulation_steps = max(1, gradient_accumulation_steps)
        self._validation_frequency = max(1, validation_frequency)
        self._early_stopping_patience = early_stopping_patience
        self._early_stopping_min_delta = early_stopping_min_delta
        self._log_every_n_steps = log_every_n_steps

        # Automatic Mixed Precision GradScaler
        self._scaler = torch.amp.GradScaler(self._device.type, enabled=self._mixed_precision)

        # State tracker
        self._state = TrainerState()

    def _train_epoch(self) -> float:
        """
        Execute one complete training epoch.

        Returns
        -------
        float
            Average training loss for the epoch.
        """

        set_train_mode(self._model)

        running_loss = 0.0
        total_batches = len(self._train_loader)

        self._optimizer.zero_grad(set_to_none=True)

        for batch in self._train_loader:
            images = move_to_device(batch.image, self._device)
            if self._device.type == "cuda":
                images = images.to(memory_format=torch.channels_last)
            targets = move_to_device(batch.label, self._device)

            step_output = train_step(
                model=self._model,
                optimizer=self._optimizer,
                criterion=self._criterion,
                images=images,
                targets=targets,
                scaler=self._scaler,
                mixed_precision=self._mixed_precision,
                device_type=self._device.type,
            )

            import math
            if math.isnan(step_output.loss):
                raise ValueError(f"NaN loss detected at step {self._state.global_step}!")

            running_loss += step_output.loss
            self._state.increment_step()

            if (
                self._log_every_n_steps > 0
                and self._state.global_step % self._log_every_n_steps == 0
            ):
                print(
                    f"[Train] Step {self._state.global_step} | "
                    f"Loss: {step_output.loss:.4f}",
                    flush=True,
                )

        average_loss = calculate_average_loss(running_loss, total_batches)
        self._state.train_loss = average_loss
        return average_loss

    def _validation_epoch(self) -> EpochOutput:
        """
        Execute one complete validation epoch.

        Returns
        -------
        EpochOutput
            Validation loss and evaluation metrics.
        """

        set_eval_mode(self._model)

        running_loss = 0.0
        total_batches = len(self._validation_loader)

        logits_history: list[torch.Tensor] = []
        targets_history: list[torch.Tensor] = []

        with torch.no_grad():
            for batch in self._validation_loader:
                images = move_to_device(batch.image, self._device)
                if self._device.type == "cuda":
                    images = images.to(memory_format=torch.channels_last)
                targets = move_to_device(batch.label, self._device)

                with autocast(device_type=self._device.type, enabled=self._mixed_precision):
                    logits, step_output = validation_step(
                        model=self._model,
                        criterion=self._criterion,
                        images=images,
                        targets=targets,
                    )

                running_loss += step_output.loss
                logits_history.append(logits.detach().cpu())
                targets_history.append(targets.detach().cpu())

        average_loss = calculate_average_loss(running_loss, total_batches)
        metrics: dict[str, float] = {}

        if logits_history:
            logits_tensor = torch.cat(logits_history, dim=0)
            targets_tensor = torch.cat(targets_history, dim=0)

            for metric_name, metric_fn in self._metrics.items():
                metric_value = metric_fn(logits_tensor, targets_tensor)
                if isinstance(metric_value, torch.Tensor):
                    metric_value = metric_value.item()
                metrics[metric_name] = float(metric_value)

        self._state.validation_loss = average_loss
        return EpochOutput(loss=average_loss, metrics=metrics)

    def fit(
        self,
        epochs: int,
        monitor: str = "val_loss",
    ) -> TrainingOutput:
        """
        Run the complete training process for a specified number of epochs.

        Parameters
        ----------
        epochs : int
            Total number of training epochs.
        monitor : str, default="val_loss"
            Metric monitored for checkpointing and early stopping.

        Returns
        -------
        TrainingOutput
            Final training summary.
        """

        is_loss_monitor = (monitor == "val_loss" or "loss" in monitor.lower())
        if is_loss_monitor and self._state.best_metric == float("-inf"):
            self._state.best_metric = float("inf")

        epochs_without_improvement = 0

        for _ in range(epochs):
            self._state.next_epoch()

            train_loss = self._train_epoch()
            validation_output: EpochOutput | None = None

            if self._state.current_epoch % self._validation_frequency == 0:
                validation_output = self._validation_epoch()
                self._state.add_epoch_result(validation_output)

                if is_loss_monitor:
                    monitored_value = validation_output.loss
                    improved = (
                        monitored_value
                        < self._state.best_metric - self._early_stopping_min_delta
                    )
                else:
                    monitored_value = validation_output.metrics.get(monitor)
                    if monitored_value is None:
                        monitored_value = validation_output.loss
                        improved = (
                            monitored_value
                            < self._state.best_metric - self._early_stopping_min_delta
                        )
                    else:
                        improved = (
                            monitored_value
                            > self._state.best_metric + self._early_stopping_min_delta
                        )

                if improved:
                    self._state.best_metric = monitored_value
                    self._state.best_epoch = self._state.current_epoch
                    epochs_without_improvement = 0

                    self._checkpoint_manager.save_best(
                        model=self._model,
                        optimizer=self._optimizer,
                        scheduler=self._scheduler,
                        trainer_state=self._state,
                    )
                else:
                    epochs_without_improvement += 1

                if epochs_without_improvement >= self._early_stopping_patience:
                    print(f"\nEarly stopping triggered at epoch {self._state.current_epoch}")
                    break

            if self._scheduler is not None:
                if isinstance(self._scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    if validation_output is not None:
                        self._scheduler.step(validation_output.loss)
                else:
                    self._scheduler.step()

            self._checkpoint_manager.save_latest(
                model=self._model,
                optimizer=self._optimizer,
                scheduler=self._scheduler,
                trainer_state=self._state,
            )

            current_lr = self._optimizer.param_groups[0]["lr"]
            if validation_output is not None:
                print(
                    f"[Epoch {self._state.current_epoch}/{epochs}] "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {validation_output.loss:.4f} | "
                    f"LR: {current_lr:.6f}"
                )
                for metric_name, metric_value in validation_output.metrics.items():
                    print(f"    {metric_name}: {metric_value:.4f}")

        return TrainingOutput(
            best_epoch=self._state.best_epoch,
            best_metric=self._state.best_metric,
            epochs_completed=self._state.current_epoch,
        )

    def test(
        self,
        test_loader: DataLoader,
    ) -> EpochOutput:
        """
        Evaluate the trained model on a test dataset.

        Parameters
        ----------
        test_loader : DataLoader
            Test dataset dataloader.

        Returns
        -------
        EpochOutput
            Test loss and evaluation metrics.
        """

        set_eval_mode(self._model)

        running_loss = 0.0
        total_batches = len(test_loader)

        logits_history: list[torch.Tensor] = []
        targets_history: list[torch.Tensor] = []

        with torch.no_grad():
            for batch in test_loader:
                images = move_to_device(batch.image, self._device)
                if self._device.type == "cuda":
                    images = images.to(memory_format=torch.channels_last)
                targets = move_to_device(batch.label, self._device)

                with torch.amp.autocast(device_type=self._device.type, enabled=self._mixed_precision):
                    logits, step_output = validation_step(
                        model=self._model,
                        criterion=self._criterion,
                        images=images,
                        targets=targets,
                    )

                running_loss += step_output.loss
                logits_history.append(logits.detach().cpu())
                targets_history.append(targets.detach().cpu())

        average_loss = calculate_average_loss(running_loss, total_batches)
        metrics: dict[str, float] = {}

        if logits_history:
            logits_tensor = torch.cat(logits_history, dim=0)
            targets_tensor = torch.cat(targets_history, dim=0)

            for metric_name, metric_fn in self._metrics.items():
                metric_value = metric_fn(logits_tensor, targets_tensor)
                if isinstance(metric_value, torch.Tensor):
                    metric_value = metric_value.item()
                metrics[metric_name] = float(metric_value)

        print("\n" + "=" * 40)
        print("TEST EVALUATION RESULTS")
        print("=" * 40)
        print(f"Test Loss : {average_loss:.4f}")
        for metric_name, metric_value in metrics.items():
            print(f"{metric_name:10s}: {metric_value:.4f}")
        print("=" * 40)

        return EpochOutput(loss=average_loss, metrics=metrics)

    def save_checkpoint(self, path: Path) -> None:
        """Save model, optimizer, scheduler, and trainer state to specified path."""
        try:
            self._checkpoint_manager.save(
                checkpoint_path=path,
                model=self._model,
                optimizer=self._optimizer,
                scheduler=self._scheduler,
                trainer_state=self._state,
            )
        except Exception as error:
            raise CheckpointError(f"Failed to save checkpoint: {path}") from error

    def load_checkpoint(self, path: Path) -> None:
        """Load model, optimizer, scheduler, and trainer state from specified path."""
        try:
            self._state = self._checkpoint_manager.load(
                checkpoint_path=path,
                model=self._model,
                optimizer=self._optimizer,
                scheduler=self._scheduler,
                device=self._device,
            )
        except Exception as error:
            raise CheckpointError(f"Failed to load checkpoint: {path}") from error

    @property
    def model(self) -> nn.Module:
        """Return the training model."""
        return self._model

    @property
    def optimizer(self) -> Optimizer:
        """Return optimizer."""
        return self._optimizer

    @property
    def scheduler(self) -> LRScheduler | None:
        """Return scheduler."""
        return self._scheduler

    @property
    def state(self) -> TrainerState:
        """Return trainer state."""
        return self._state

    @property
    def device(self) -> torch.device:
        """Return execution device."""
        return self._device
