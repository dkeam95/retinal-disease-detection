"""
Dataset module for retinal disease detection.
"""

# Core PyTorch Dataset implementation and file parsing utilities
from dataset.dataset import RetinalDataset  # Custom PyTorch Dataset class for loading retinal images
from dataset.parser import load_annotations  # Utility function to read and parse annotation text/CSV files

# Strongly typed dataclasses representing dataset items
from dataset.types import (
    AnnotationRecord,  # Dataclass storing image filename and corresponding label info
    DataSample,        # Dataclass containing loaded image tensor and target tensor
)

# Domain-specific dataset exception hierarchy
from dataset.exceptions import (
    DatasetError,                 # Base exception class for dataset-related issues
    DatasetNotFoundError,        # Raised when the main dataset directory is missing
    AnnotationFileNotFoundError, # Raised when the annotations file cannot be located
    InvalidAnnotationError,      # Raised when annotation entries are corrupted or misformatted
    ImageLoadingError,           # Raised when an image file fails to read or decode
    EmptyDatasetError,           # Raised when no valid samples are found in the dataset
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