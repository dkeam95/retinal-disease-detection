"""
PyTorch dataset implementation for retinal disease detection.

This dataset is designed to load retinal images and their corresponding labels
from disk and provide them to the PyTorch training pipeline.
"""

from __future__ import annotations  # Enables modern type hints (Python 3.7+)

from pathlib import Path            # Object-oriented filesystem path navigation
from typing import cast             # Type hint utility to inform static type checkers

import cv2                          # OpenCV library for fast image I/O and color space conversions
import numpy as np                  # Fundamental package for array manipulation
from numpy.typing import NDArray    # Type hint for NumPy array shapes and dtypes

from torch.utils.data import Dataset  # PyTorch base class for custom dataset loaders

from common.config.types import DatasetConfig     # Dataclass holding dataset configuration properties
from dataset.exceptions import ImageLoadingError  # Domain-specific exception raised when OpenCV fails
from dataset.parser import load_annotations       # Utility function to parse annotation file into records
from dataset.types import DataSample              # Typed object containing loaded sample data


class RetinalDataset(Dataset[DataSample]):
    """
    PyTorch dataset for retinal disease classification.
    """

    def __init__(self, config: DatasetConfig) -> None:
        """
        Initialize the dataset.

        Parameters
        ----------
        config : DatasetConfig
            Dataset configuration.
        """

        # Parse and cache annotation records using absolute/relative resolved paths
        self._annotations = load_annotations(
            annotation_file=config.path / config.annotation_file,
            image_directory=config.path / config.image_directory,
        )

    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.
        """

        # Returns the total count of parsed dataset samples
        return len(self._annotations)

    def _load_image(
        self,
        image_path: Path,
    ) -> NDArray[np.uint8]:
        """
        Load an image from disk.

        Parameters
        ----------
        image_path : Path
            Path to the image file.

        Returns
        -------
        NDArray[np.uint8]
            RGB image loaded from disk.

        Raises
        ------
        ImageLoadingError
            If the image cannot be loaded.
        """

        # Ensure image file exists before attempting to read it
        if not image_path.exists():
            raise ImageLoadingError(
                f"Image file does not exist: {image_path}"
            )

        # Read image file using OpenCV in standard 3-channel color mode
        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_COLOR,
        )

        # Check if cv2 returned None (file missing, corrupt, or unsupported format)
        if image is None:
            raise ImageLoadingError(
                f"Failed to load image: {image_path}"
            )

        # Validate image dimensionality
        if image.ndim != 3:
            raise ImageLoadingError(
                f"Invalid image dimensions: {image.shape}"
            )

        # Validate image channel count
        if image.shape[2] != 3:
            raise ImageLoadingError(
                f"Expected 3-channel image but got shape {image.shape}"
            )

        # OpenCV loads images in BGR format by default; convert to standard RGB
        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

        # Cast array to explicit uint8 NDArray for type checking safety
        return cast(
            NDArray[np.uint8],
            image,
        )

    def __getitem__(
        self,
        index: int,
    ) -> DataSample:
        """
        Get a dataset sample.

        Parameters
        ----------
        index : int
            Sample index.

        Returns
        -------
        DataSample
            Loaded dataset sample.
        """

        # Explicitly validate bounds to prevent silent negative or out-of-range indexing bugs
        if not 0 <= index < len(self):
            raise IndexError(
                f"Dataset index out of range: {index}"
            )

        # Retrieve target annotation metadata record
        annotation = self._annotations[index]

        # Load RGB image array from disk using verified path
        image = self._load_image(
            annotation.image_path,
        )

        # Construct and return strongly typed DataSample object
        return DataSample(
            image_path=annotation.image_path,
            image=image,
            label=annotation.label,
        )