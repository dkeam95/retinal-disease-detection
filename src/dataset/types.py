"""
Type definitions for the dataset module.

This module contains immutable data structures shared across the
dataset pipeline.
"""

from __future__ import annotations  # Enables modern type hints (Python 3.7+)

from dataclasses import (
    dataclass,  # Decorator to automatically generate special methods for classes
)
from pathlib import Path  # Object-oriented filesystem path navigation

import numpy as np  # Fundamental package for array manipulation
from numpy.typing import NDArray  # Type hint for NumPy array shapes and dtypes


@dataclass(frozen=True, slots=True)
class AnnotationRecord:
    """
    Represents a single dataset annotation.
    """

    image_path: Path  # Resolved path to the target image file
    label: int        # Non-negative target class label identifier


@dataclass(frozen=True, slots=True)
class DataSample:
    """
    Represents a single dataset sample.
    """

    image_path: Path           # Resolved path to the target image file
    image: NDArray[np.uint8]   # Loaded 3-channel RGB image tensor as NumPy array
    label: int                 # Non-negative target class label identifier
