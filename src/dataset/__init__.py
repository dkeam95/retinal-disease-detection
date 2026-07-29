"""
Dataset module for retinal disease detection.
"""

# Core PyTorch Dataset implementation and file parsing utilities
from dataset.dataset import (
    RetinalDataset,  # Custom PyTorch Dataset class for loading retinal images
)

# Domain-specific dataset exception hierarchy
from dataset.exceptions import (
    AnnotationFileNotFoundError,  # Raised when the annotations file cannot be located
    DatasetError,  # Base exception class for dataset-related issues
    DatasetNotFoundError,  # Raised when the main dataset directory is missing
    EmptyDatasetError,  # Raised when no valid samples are found in the dataset
    ImageLoadingError,  # Raised when an image file fails to read or decode
    InvalidAnnotationError,  # Raised when annotation entries are corrupted or misformatted
)
from dataset.parser import (
    load_annotations,  # Utility function to read and parse annotation text/CSV files
)

# Strongly typed dataclasses representing dataset items
from dataset.types import (
    AnnotationRecord,  # Dataclass storing image filename and corresponding label info
    DataSample,  # Dataclass containing loaded image tensor and target tensor
)

# Explicitly define public API exports for module imports
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
