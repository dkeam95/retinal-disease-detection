"""Unit tests for RetinalDataset implementation."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from common.config.types import DatasetConfig
from dataset.dataset import RetinalDataset
from dataset.exceptions import ImageLoadingError


@pytest.fixture
def dummy_dataset_env(tmp_path: Path) -> DatasetConfig:
    """Fixture that creates dummy images, annotation file, and DatasetConfig."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()

    # Create dummy RGB image (10x10x3) and save via cv2
    img1_path = img_dir / "test1.png"
    dummy_img = np.zeros((10, 10, 3), dtype=np.uint8)
    dummy_img[:, :] = (255, 0, 0)  # Blue color in BGR
    cv2.imwrite(str(img1_path), dummy_img)

    # Create annotation file
    annot_path = tmp_path / "train.txt"
    annot_path.write_text("test1.png 2\n", encoding="utf-8")

    return DatasetConfig(
        path=tmp_path,
        annotation_file="train.txt",
        image_directory="images",
        num_classes=5,
    )


def test_retinal_dataset_init_and_len(dummy_dataset_env: DatasetConfig) -> None:
    """Test proper dataset initialization and length reporting."""
    dataset = RetinalDataset(dummy_dataset_env)
    assert len(dataset) == 1


def test_retinal_dataset_getitem_success(dummy_dataset_env: DatasetConfig) -> None:
    """Test retrieving a valid sample with BGR to RGB conversion."""
    dataset = RetinalDataset(dummy_dataset_env)
    sample = dataset[0]

    assert sample.label == 2
    assert sample.image.shape == (10, 10, 3)
    assert sample.image.dtype == np.uint8
    # In cv2 the write was (255, 0, 0) BGR -> in RGB it should be (0, 0, 255)
    assert np.array_equal(sample.image[0, 0], np.array([0, 0, 255], dtype=np.uint8))


def test_retinal_dataset_getitem_out_of_bounds(dummy_dataset_env: DatasetConfig) -> None:
    """Test that IndexError is raised for out-of-range indices."""
    dataset = RetinalDataset(dummy_dataset_env)

    with pytest.raises(IndexError):
        _ = dataset[5]

    with pytest.raises(IndexError):
        _ = dataset[-1]


def test_load_image_missing_file(dummy_dataset_env: DatasetConfig, tmp_path: Path) -> None:
    """Test ImageLoadingError when the target image file does not exist on disk."""
    dataset = RetinalDataset(dummy_dataset_env)
    missing_file = tmp_path / "images" / "non_existent.jpg"

    with pytest.raises(ImageLoadingError) as exc_info:
        dataset._load_image(missing_file)
    assert "Image file does not exist" in str(exc_info.value)


def test_load_image_corrupt_or_invalid_file(dummy_dataset_env: DatasetConfig, tmp_path: Path) -> None:
    """Test ImageLoadingError when cv2 fails to decode the image file."""
    dataset = RetinalDataset(dummy_dataset_env)
    corrupt_file = tmp_path / "images" / "corrupt.jpg"
    corrupt_file.write_text("not an image content", encoding="utf-8")

    with pytest.raises(ImageLoadingError) as exc_info:
        dataset._load_image(corrupt_file)
    assert "Failed to load image" in str(exc_info.value)
