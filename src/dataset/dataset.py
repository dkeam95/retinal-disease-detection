"""Custom  PyTorch Dataset implementation for retinal disease detection."""

import csv
from pathlib import Path
from torch.utils.data import Dataset

from common.config.types import DatasetConfig
from dataset.exceptions import (
    DatasetNotFoundError,
    EmptyDatasetError,
    InvalidMetadataError,
)
from dataset.types import DataSample


class RetinalDataset:
    """Dataset class for loading and parsing retinal fundus images."""

    def __init__(self, config: DatasetConfig) -> None:
        """Initialize the dataset with configuration parameters.

        Args:
            config: Configuration object containing data paths and metadata.
        """
        self._config = config
        self._samples: list[DataSample] = []
        self._load_metadata()

    def __len__(self) -> int:
        """Return the total number of samples in the dataset.

        Returns:
            The total sample count.
        """
        return len(self._samples)

    def __getitem__(self, index: int) -> tuple[str, int]:
        """Fetch a single data sample by its index.

        Args:
            index: The index of the item to retrieve.

        Returns:
            A tuple containing the image path as a string and the integer label.
        """
        sample = self._samples[index]
        return str(sample.image_path), sample.label

    def _load_metadata(self) -> None:
        """Parse the metadata file and populate the internal samples list.

        Raises:
            DatasetNotFoundError: If the dataset path or metadata does not exist.
            InvalidMetadataError: If the metadata layout is unreadable or corrupted.
            EmptyDatasetError: If no valid image samples are loaded.
        """

        dataset_root = Path(self._config.path)
        metadata_file = dataset_root / "metadata.csv"

        if not dataset_root.exists() or not metadata_file.exists():
            raise DatasetNotFoundError(
                f"Dataset root or metadata.csv not found at: {dataset_root}"
            )

        try:
            with open(metadata_file, mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)

                # Check for required SCV columns
                if reader.fieldnames is None or not {"image_id", "label"}.issubset(
                    reader.fieldnames
                ):
                    raise InvalidMetadataError(
                        "Metadata CSV must contain 'image_id' and 'label' columns."
                    )

                for row_idx, row in enumerate(reader, start=1):
                    image_id = row.get("image_id")
                    label_str = row.get("label")

                    if not image_id or label_str is None:
                        raise InvalidMetadataError(
                            f"Missing values at row {row_idx} in metadata.csv"
                        )

                    try:
                        label = int(label_str)
                    except ValueError as err:
                        raise InvalidMetadataError(
                            f"Invalid label format at row {row_idx}: {label_str}"
                        ) from err

                    image_path = dataset_root / "images" / image_id

                    # Store sample if the image file exists on disk
                    if image_path.exists():
                        self._samples.append(
                            DataSample(image_path=image_path, label=label)
                        )

        except (IOError, OSError) as err:
            raise InvalidMetadataError(f"Failed to read metadata file: {err}") from err

        if not self._samples:
            raise EmptyDatasetError(
                f"No valid image samples discovered in: {dataset_root / 'images'}"
            )
