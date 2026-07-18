"""Pytorch dataset implementation for retinal disease detection."""
from __future__ import annotations

from torch.utils.data import Dataset

from pathlib import Path
from typing import cast

import cv2
import numpy as np

from dataset.parser import load_annotations
from numpy.typing import NDArray
from common.config.types import DatasetConfig
from dataset.types import DataSample
from dataset.exceptions import ImageLoadingError, EmptyDatasetError

class RetinalDataset(Dataset[DataSample]):
    """PyTorch dataset for retinal disease classification."""


    def __init__(self, config: DatasetConfig) -> None:
        """Initialize the dataset.

        Parameters:
        config: DatasetConfig
            Dataset configuration
        """

        self._config = config
        self._annotations = load_annotations(
            annotation_file=(self._config.path / self._config.annotation_file),
            image_directory=(self._config.path / self._config.image_directory)
        )

        if not self._annotations:
            raise EmptyDatasetError(
                f"Dataset contains no annotation records"
            )
        


    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self._annotations)


    def _load_image(self, image_path: Path) -> NDArray[np.uint8]:
        """Load an image from disk.
        
           Parameters:
           image_path: Path
               Path to the image file
           
           Returns:
           NDArray[np.uint8]
               Loaded image as a NumPy array
           
           Raises:
           ImageLoadingError
               If the image cannot be loaded
        """

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

        if image is None:
            raise ImageLoadingError(
                f"Failed to load image: {image_path}"
            )

        return cast(NDArray[np.uint8], image)


    def __getitem__(self, index: int) -> DataSample:
        """Get a dataset sample by index.
        
           Parameters:
           index: int
               Index of the sample to retrieve
           
           Returns:
           DataSample
               Dataset sample containing image and label
        """
        annotation = self._annotations[index]

        image = self._load_image(annotation.image_path)

        return DataSample(
            image=image,
            label=annotation.label,
            image_path=annotation.image_path
        )