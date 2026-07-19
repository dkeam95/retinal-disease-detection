"""
PyTorch dataset implementation for retinal disease detection.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray

from torch.utils.data import Dataset

from common.config.types import DatasetConfig
from dataset.exceptions import ImageLoadingError
from dataset.parser import load_annotations
from dataset.types import DataSample


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

        self._annotations = load_annotations(
            annotation_file=config.path / config.annotation_file,
            image_directory=config.path / config.image_directory,
        )

    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.
        """
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

        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_COLOR,
        )

        if image is None:
            raise ImageLoadingError(
                f"Failed to load image: {image_path}"
            )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

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

        if not 0 <= index < len(self):
            raise IndexError(
                f"Dataset index out of range: {index}"
            )

        annotation = self._annotations[index]

        image = self._load_image(
            annotation.image_path,
        )

        return DataSample(
            image_path=annotation.image_path,
            image=image,
            label=annotation.label,
        )