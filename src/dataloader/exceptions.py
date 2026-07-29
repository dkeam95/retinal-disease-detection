"""
Custom exceptions for the dataloader module.

This module defines the custom exception hierarchy used throughout the dataloader package,
ensuring consistent error handling during PyTorch DataLoader construction, worker process
configuration, batch size validation, and sampler initialization.
"""

from __future__ import annotations  # Enables modern type hints (Python 3.7+)


class DataLoaderError(Exception):
    """Base exception for all dataloader-related errors."""


class InvalidBatchSizeError(DataLoaderError):
    """Raised when the configured batch size is invalid (e.g., non-positive integer)."""


class InvalidNumWorkersError(DataLoaderError):
    """Raised when the configured number of data loading worker processes is invalid."""


class InvalidSamplerError(DataLoaderError):
    """Raised when an unsupported or misconfigured sampler is provided."""


class DataLoaderBuilderError(DataLoaderError):
    """Raised when a PyTorch DataLoader instance cannot be successfully constructed."""
