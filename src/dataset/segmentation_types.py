"""Type definitions for retinal lesion segmentation.

This module contains immutable data structures shared across the
lesion segmentation pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class SegmentationAnnotationRecord:
    """Represents one retinal image together with all lesion masks."""

    image_path: Path

    ex_mask_path: Path | None  # Exudate mask
    he_mask_path: Path | None  # Hemorrhage mask
    ma_mask_path: Path | None  # Microaneurysm mask
    se_mask_path: Path | None  # Cotton wool spot mask


@dataclass(frozen=True, slots=True)
class SegmentationDataSample:
    """Represents one segmentation sample."""

    image_path: Path

    image: NDArray[np.uint8]

    mask: NDArray[np.uint8]
