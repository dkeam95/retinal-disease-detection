"""
Dataset module for retinal disease detection.
"""

from dataset.dataset import RetinalDataset
from dataset.parser import load_annotations

from dataset.types import (
    AnnotationRecord,
    DataSample,
)

from dataset.exceptions import (
    DatasetError,
    DatasetNotFoundError,
    AnnotationFileNotFoundError,
    InvalidAnnotationError,
    ImageLoadingError,
    EmptyDatasetError,
)

__all__ = [
    "RetinalDataset",
    "load_annotations",
    "AnnotationRecord",
    "DataSample",
    "DatasetError",
    "DatasetNotFoundError",
    "AnnotationFileNotFoundError",
    "InvalidAnnotationError",
    "ImageLoadingError",
    "EmptyDatasetError",
]