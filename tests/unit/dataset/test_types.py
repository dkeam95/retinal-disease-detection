"""Unit tests for dataset module type definitions."""

from pathlib import Path
import numpy as np
import pytest

from src.dataset.types import AnnotationRecord, DataSample


def test_annotation_record_creation(tmp_path: Path) -> None:
    """Test valid creation and attribute access of AnnotationRecord."""
    img_path = tmp_path / "sample.jpg"
    record = AnnotationRecord(image_path=img_path, label=1)

    assert record.image_path == img_path
    assert record.label == 1


def test_annotation_record_immutability(tmp_path: Path) -> None:
    """Test that AnnotationRecord is frozen and cannot be modified."""
    record = AnnotationRecord(image_path=tmp_path / "sample.jpg", label=0)

    with pytest.raises(AttributeError):
        record.label = 2  # type: ignore[misc]


def test_data_sample_creation(tmp_path: Path) -> None:
    """Test valid creation and array properties of DataSample."""
    img_path = tmp_path / "sample.jpg"
    dummy_image = np.zeros((224, 224, 3), dtype=np.uint8)

    sample = DataSample(image_path=img_path, image=dummy_image, label=2)

    assert sample.image_path == img_path
    assert sample.label == 2
    assert sample.image.shape == (224, 224, 3)
    assert sample.image.dtype == np.uint8


def test_data_sample_immutability(tmp_path: Path) -> None:
    """Test that DataSample is frozen."""
    dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
    sample = DataSample(image_path=tmp_path / "sample.jpg", image=dummy_image, label=0)

    with pytest.raises(AttributeError):
        sample.image_path = tmp_path / "other.jpg"  # type: ignore[misc]