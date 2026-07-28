"""
Checkpoint manager module.

This module provides a high-level controller interface for managing, saving,
restoring, and organizing model checkpoints throughout the training lifecycle.
"""

from __future__ import annotations  # Enables modern type hints (|)

from pathlib import Path  # Object-oriented filesystem path management
from typing import Any  # Dynamic typing for state dictionaries

import torch  # PyTorch deep learning framework
from torch import nn  # Neural network module base class
from torch.optim import Optimizer  # Optimization algorithms
from torch.optim.lr_scheduler import LRScheduler  # Learning rate scheduler base class

from checkpoint.exceptions import (
    CheckpointLoadError,
    CheckpointNotFoundError,
    CheckpointSaveError,
)
from checkpoint.utils import (
    build_checkpoint_filename,
    create_checkpoint_directory,
    find_latest_checkpoint,
    validate_checkpoint,
)
from trainer.state import TrainerState  # Training progress tracking data structure


class CheckpointManager:
    """
    High-level controller handling checkpoint serialization, deserialization,
    and filesystem management.
    """

    def __init__(
        self,
        checkpoint_directory: Path,
    ) -> None:
        """
        Initialize the checkpoint manager and ensure storage directory exists.

        Parameters
        ----------
        checkpoint_directory : Path
            Directory path where all checkpoint files will be written and read.
        """

        self._checkpoint_directory = checkpoint_directory

        # Create output directory automatically if it does not exist
        create_checkpoint_directory(
            self._checkpoint_directory,
        )

    def _build_checkpoint(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: LRScheduler | None,
        trainer_state: TrainerState,
    ) -> dict[str, Any]:
        """
        Construct a serializable dictionary containing all system state snapshots.

        Parameters
        ----------
        model : nn.Module
            Neural network model instance.
        optimizer : Optimizer
            Optimizer instance.
        scheduler : LRScheduler | None
            Learning rate scheduler instance or None.
        trainer_state : TrainerState
            State container with progress metrics.

        Returns
        -------
        dict[str, Any]
            Dictionary containing state dicts and metadata ready for saving.
        """

        return {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": (
                scheduler.state_dict()
                if scheduler is not None
                else None
            ),
            "trainer_state": trainer_state,
            "metadata": {
                "epoch": trainer_state.current_epoch,
                "best_epoch": trainer_state.best_epoch,
                "best_metric": trainer_state.best_metric,
                "global_step": trainer_state.global_step,
            },
        }

    def _restore_checkpoint(
        self,
        checkpoint: dict[str, Any],
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: LRScheduler | None,
    ) -> TrainerState:
        """
        Restore state dictionaries into their respective PyTorch objects.

        Parameters
        ----------
        checkpoint : dict[str, Any]
            Loaded checkpoint state dictionary.
        model : nn.Module
            Target model instance to load parameters into.
        optimizer : Optimizer
            Target optimizer instance to load state into.
        scheduler : LRScheduler | None
            Target learning rate scheduler instance to load state into.

        Returns
        -------
        TrainerState
            Restored trainer progress and metric state instance.
        """

        # Restore parameters for model and optimizer
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])

        # Restore scheduler state if both instance and saved state exist
        if scheduler is not None and checkpoint.get("scheduler_state") is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state"])

        return checkpoint["trainer_state"]

    def _best_checkpoint_path(self) -> Path:
        """
        Get the fixed file path for the best performing checkpoint alias.

        Returns
        -------
        Path
            Path pointing to 'best.pt' in the checkpoint directory.
        """

        return self._checkpoint_directory / "best.pt"

    def _latest_checkpoint_path(self) -> Path:
        """
        Get the fixed file path for the latest training checkpoint alias.

        Returns
        -------
        Path
            Path pointing to 'latest.pt' in the checkpoint directory.
        """

        return self._checkpoint_directory / "latest.pt"

    def save(
        self,
        checkpoint_path: Path,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: LRScheduler | None,
        trainer_state: TrainerState,
    ) -> Path:
        """
        Save model and trainer states to a specific checkpoint file path safely.

        Parameters
        ----------
        checkpoint_path : Path
            Destination file path.
        model : nn.Module
            Model instance to serialize.
        optimizer : Optimizer
            Optimizer instance to serialize.
        scheduler : LRScheduler | None
            Optional learning rate scheduler instance to serialize.
        trainer_state : TrainerState
            Current trainer execution state.

        Returns
        -------
        Path
            Path where the checkpoint was written.

        Raises
        ------
        CheckpointSaveError
            If write permissions fail or serialization encounters an error.
        """

        # Assemble state dictionary package
        checkpoint = self._build_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            trainer_state=trainer_state,
        )

        # Temporary file path for atomic write protection
        temp_path = checkpoint_path.with_suffix(".tmp")

        try:
            # Write to temporary file first to prevent corrupting existing checkpoints
            torch.save(checkpoint, temp_path)
            # Atomically replace destination path with temporary file
            temp_path.replace(checkpoint_path)

        except (OSError, RuntimeError) as error:
            # Clean up temporary file if it was created
            if temp_path.exists():
                temp_path.unlink()

            raise CheckpointSaveError(
                f"Failed to save checkpoint to path: {checkpoint_path}"
            ) from error

        return checkpoint_path

    def save_epoch(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: LRScheduler | None,
        trainer_state: TrainerState,
    ) -> Path:
        """
        Save an epoch-indexed checkpoint using the state's epoch and metric values.

        Returns
        -------
        Path
            Path to the saved epoch checkpoint file.
        """

        # Generate standard filename based on current epoch index and metric
        filename = build_checkpoint_filename(
            epoch=trainer_state.current_epoch,
            metric=trainer_state.best_metric,
        )
        target_path = self._checkpoint_directory / filename

        return self.save(
            checkpoint_path=target_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            trainer_state=trainer_state,
        )

    def save_best(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: LRScheduler | None,
        trainer_state: TrainerState,
    ) -> Path:
        """
        Save state to the designated 'best.pt' file path.

        Returns
        -------
        Path
            Path to the saved best checkpoint file.
        """

        return self.save(
            checkpoint_path=self._best_checkpoint_path(),
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            trainer_state=trainer_state,
        )

    def save_latest(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: LRScheduler | None,
        trainer_state: TrainerState,
    ) -> Path:
        """
        Save state to the designated 'latest.pt' file path.

        Returns
        -------
        Path
            Path to the saved latest checkpoint file.
        """

        return self.save(
            checkpoint_path=self._latest_checkpoint_path(),
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            trainer_state=trainer_state,
        )

    def load(
        self,
        checkpoint_path: Path,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: LRScheduler | None,
        device: torch.device | str = "cpu",
    ) -> TrainerState:
        """
        Load a training checkpoint into provided model, optimizer, and scheduler.

        Parameters
        ----------
        checkpoint_path : Path
            File path of the checkpoint to restore.
        model : nn.Module
            Model target for state loading.
        optimizer : Optimizer
            Optimizer target for state loading.
        scheduler : LRScheduler | None
            Scheduler target for state loading.
        device : torch.device | str, default="cpu"
            Compute device target to map tensors during deserialization.

        Returns
        -------
        TrainerState
            Restored trainer state instance.

        Raises
        ------
        CheckpointLoadError
            If file reading or state restoration fails.
        """

        # Ensure target checkpoint file exists and is valid
        validate_checkpoint(checkpoint_path)

        try:
            # weights_only=False is required to deserialize custom dataclass TrainerState
            checkpoint = torch.load(
                checkpoint_path,
                map_location=device,
                weights_only=False,
            )

        except (OSError, RuntimeError, TypeError) as error:
            raise CheckpointLoadError(
                f"Failed to load checkpoint file: {checkpoint_path}"
            ) from error

        return self._restore_checkpoint(
            checkpoint=checkpoint,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
        )

    def load_best(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: LRScheduler | None,
        device: torch.device | str = "cpu",
    ) -> TrainerState:
        """
        Load the 'best.pt' checkpoint if available.

        Raises
        ------
        CheckpointNotFoundError
            If 'best.pt' does not exist in the checkpoint directory.
        """

        checkpoint_path = self._best_checkpoint_path()

        if not checkpoint_path.exists():
            raise CheckpointNotFoundError(
                f"Best checkpoint not found: {checkpoint_path}"
            )

        return self.load(
            checkpoint_path=checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
        )

    def load_latest(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: LRScheduler | None,
        device: torch.device | str = "cpu",
    ) -> TrainerState:
        """
        Load the 'latest.pt' or most recent epoch checkpoint available.

        Raises
        ------
        CheckpointNotFoundError
            If no valid checkpoint files exist in the directory.
        """

        # First try loading 'latest.pt' alias
        latest_path = self._latest_checkpoint_path()

        if latest_path.exists():
            return self.load(
                checkpoint_path=latest_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
            )

        # Fallback to searching for the most recently modified epoch file
        found_path = find_latest_checkpoint(self._checkpoint_directory)

        if found_path is None:
            raise CheckpointNotFoundError(
                f"No checkpoint files found in directory: {self._checkpoint_directory}"
            )

        return self.load(
            checkpoint_path=found_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
        )