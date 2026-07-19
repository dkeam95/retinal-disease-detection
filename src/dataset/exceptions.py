"""
Custom exceptions for the dataset module.

This module defines all dataset-specific exceptions used during
dataset initialization and sample loading.
"""


class DatasetError(Exception):
    """Base exception for all dataset-related errors."""


class DatasetNotFoundError(DatasetError):
    """Raised when the dataset directory does not exist."""


class AnnotationFileNotFoundError(DatasetError):
    """Raised when the annotation file cannot be found."""


class InvalidAnnotationError(DatasetError):
    """Raised when an annotation file is invalid."""


class ImageLoadingError(DatasetError):
    """Raised when an image cannot be loaded from disk."""


class EmptyDatasetError(DatasetError):
    """Raised when a dataset contains no samples."""