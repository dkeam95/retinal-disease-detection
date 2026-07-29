"""
Custom exceptions for the preprocessing module.

This module defines all preprocessing-specific exceptions used during
pipeline construction and image preprocessing.
"""

from __future__ import annotations  # Enables modern type hints (Python 3.7+)


class PreprocessingError(Exception):
    """Base exception for all preprocessing-related errors."""


class InvalidPreprocessingConfigError(PreprocessingError):
    """Raised when the preprocessing configuration is invalid."""


class PipelineBuildError(PreprocessingError):
    """Raised when the preprocessing pipeline cannot be constructed."""


class PipelineExecutionError(PreprocessingError):
    """Raised when the preprocessing pipeline fails to process an image."""
