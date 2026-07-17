"""Custom exceptions for the dataset module."""


class DatasetError(Exception):
    """Base exception for all dataset-related errors."""

    pass


class DatasetNotFoundError(DatasetError):
    """Raised when the dataset directory or metadata file is missing."""

    pass


class InvalidMetadataError(DatasetError):
    """Raised when the metadata file has an invalid cchema or is corrupted."""

    pass


class EmptyDatasetError(DatasetError):
    """Raised when the dataset directory contains no valid samples."""

    pass
