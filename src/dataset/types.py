"""Type definitions for the dataset module."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class DataSample:
    """Represents a single dataset sample containing an image and its label."""

    image_path: Path
    label: int
