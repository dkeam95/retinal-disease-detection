"""
Type definitions for the dataset module.

This module contains immutable data structures shared across the
dataset pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class AnnotationRecord:
    """
    Represents a single dataset annotation.
    """

    image_path: Path
    label: int


@dataclass(frozen=True, slots=True)
class DataSample:
    """
    Represents a single dataset sample.
    """

    image_path: Path
    image: NDArray[np.uint8]
    label: int