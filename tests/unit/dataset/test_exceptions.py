"""Unit tests for dataset module exceptions."""

import pytest

from src.dataset.exceptions import (
    AnnotationFileNotFoundError,
    DatasetError,
    DatasetNotFoundError,
    EmptyDatasetError,
    ImageLoadingError,
    InvalidAnnotationError,
)


def test_dataset_exceptions_inheritance() -> None:
    """Test that all custom exceptions correctly inherit from DatasetError."""
    exceptions = [
        DatasetNotFoundError,
        AnnotationFileNotFoundError,
        InvalidAnnotationError,
        ImageLoadingError,
        EmptyDatasetError,
    ]

    for exc_class in exceptions:
        assert issubclass(exc_class, DatasetError)
        assert issubclass(exc_class, Exception)


def test_dataset_exceptions_raise_and_catch() -> None:
    """Test raising custom dataset exceptions and catching them as base DatasetError."""
    with pytest.raises(DatasetError) as exc_info:
        raise DatasetNotFoundError("Dataset directory not found: /path/to/dir")

    assert "Dataset directory not found" in str(exc_info.value)
