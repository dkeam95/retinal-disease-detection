"""
Preprocessing module.
"""

# Factory functions for building Albumentations image processing pipelines
from preprocessing.pipeline import (
    build_train_pipeline,       # Builds pipeline with training augmentations and normalization
    build_validation_pipeline,  # Builds deterministic validation pipeline (resizing and normalization)
    build_test_pipeline,        # Builds deterministic test pipeline (resizing and normalization)
)

# Strongly typed dataclass for preprocessing parameters
from preprocessing.config import PreprocessingSettings  # Dataclass containing image dimensions, mean, std, and probabilities

# Domain-specific preprocessing exception hierarchy
from preprocessing.exceptions import (
    PreprocessingError,                # Base exception for preprocessing errors
    InvalidPreprocessingConfigError,   # Raised when preprocessing configuration parameters are invalid
    PipelineBuildError,                # Raised when Albumentations composition fails to build
    PipelineExecutionError,            # Raised when image transformation execution fails
)

# Explicitly define public API exports for module imports
__all__ = [
    "build_train_pipeline",
    "build_validation_pipeline",
    "build_test_pipeline",
    "PreprocessingSettings",
    "PreprocessingError",
    "InvalidPreprocessingConfigError",
    "PipelineBuildError",
    "PipelineExecutionError",
]