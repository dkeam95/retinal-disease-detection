"""Type definitions for the dataset module.

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
    """Represents a single annotation entry.

    Attributes:
        image_path: Path
            Absolute or relative path to the source image.

        label: int
            Integer class label
    """

    image_path: Path
    label: int


@dataclass(frozen=True, slots=True)
class DataSample:
    """Represents a single sample loaded from the dataset.

    Attributes:
        image: NDArray[np.uint8]
            Image loaded from disk.

        label: int
            Integer class label.

        image_path:
            Absolute or relative path to the source image.
    """

    image: NDArray[np.uint8]
    label: int
    image_path: Path
