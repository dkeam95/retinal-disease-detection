"""
Utility functions for the checkpoint module.

This module provides helper functions used by the checkpoint manager
for checkpoint file generation, validation, path creation, and discovery.
"""

from __future__ import annotations  # Enables modern type hints (|)

from pathlib import Path  # Object-oriented filesystem path utility

from checkpoint.exceptions import CheckpointNotFoundError, InvalidCheckpointError  # Custom exceptions


def build_checkpoint_filename(epoch: int, metric: float) -> str:
    """
    Build a standard checkpoint filename given epoch index and metric value.

    Parameters
    ----------
    epoch : int
        Completed training epoch number.
    metric : float
        Validation metric value achieved at this epoch.

    Returns
    -------
    str
        Generated formatted checkpoint filename (e.g., 'epoch_0001_metric_0.9500.pt').
    """

    # Format epoch with 4-digit zero-padding to support up to 9999 epochs seamlessly
    return f"epoch_{epoch:04d}_metric_{metric:.4f}.pt"


def validate_checkpoint(checkpoint_path: Path) -> None:
    """
    Validate that a given path exists and points to an actual file.

    Parameters
    ----------
    checkpoint_path : Path
        Target path to verify.

    Raises
    ------
    CheckpointNotFoundError
        If the file does not exist at the specified path.
    InvalidCheckpointError
        If the path exists but points to a directory instead of a regular file.
    """

    # Verify that the path exists on disk
    if not checkpoint_path.exists():
        raise CheckpointNotFoundError(
            f"Checkpoint file not found: {checkpoint_path}"
        )

    # Ensure the path points to a file, not a directory or symlink target
    if not checkpoint_path.is_file():
        raise InvalidCheckpointError(
            f"Checkpoint path is not a valid file: {checkpoint_path}"
        )


def find_latest_checkpoint(checkpoint_directory: Path) -> Path | None:
    """
    Find the most recently modified checkpoint file in the specified directory.

    Parameters
    ----------
    checkpoint_directory : Path
        Directory containing stored checkpoint files.

    Returns
    -------
    Path | None
        Path to the latest checkpoint file, or None if the directory does not exist or is empty.
    """

    # Return None early if directory does not exist or is not a directory
    if not checkpoint_directory.is_dir():
        return None

    # Retrieve all files matching the checkpoint pattern
    checkpoints = list(checkpoint_directory.glob("*.pt"))

    # Return None if no checkpoint files were found
    if not checkpoints:
        return None

    # Sort files by last modification timestamp (mtime) to accurately identify the newest file
    checkpoints.sort(key=lambda path: path.stat().st_mtime)

    # Return the most recently updated checkpoint
    return checkpoints[-1]


def create_checkpoint_directory(checkpoint_directory: Path) -> None:
    """
    Create the target directory for storing checkpoints if it does not already exist.

    Parameters
    ----------
    checkpoint_directory : Path
        Directory path to create.
    """

    # Create directory and parent directories if needed, ignoring error if it already exists
    checkpoint_directory.mkdir(parents=True, exist_ok=True)