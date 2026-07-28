"""
Custom exceptions for the checkpoint module.

This module defines the hierarchy of exceptions used across saving, loading,
and validating model and trainer checkpoints.
"""


class CheckpointError(Exception):
    """
    Base exception class for all checkpoint-related operations.
    """


class CheckpointNotFoundError(CheckpointError, FileNotFoundError):
    """
    Raised when a requested checkpoint file or directory does not exist on disk.
    
    Inherits from FileNotFoundError to allow standard Python exception handling.
    """


class InvalidCheckpointError(CheckpointError):
    """
    Raised when a checkpoint file is corrupted, incomplete, or fails format verification.
    """


class CheckpointSaveError(CheckpointError):
    """
    Raised when an error occurs during the checkpoint serialization process.
    """


class CheckpointLoadError(CheckpointError):
    """
    Raised when an error occurs while reading or restoring checkpoint states.
    """