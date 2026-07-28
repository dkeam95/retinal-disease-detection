"""Unit tests for dataset annotation parser."""

from pathlib import Path
import pytest

from src.dataset.exceptions import (
    AnnotationFileNotFoundError,
    EmptyDatasetError,
    InvalidAnnotationError,
)
from src.dataset.parser import load_annotations


def test_load_annotations_success(tmp_path: Path) -> None:
    """Test successful parsing of a valid annotation file."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    
    # Создаем фиктивные изображения
    (img_dir / "img1.jpg").touch()
    (img_dir / "img2.jpg").touch()

    annot_file = tmp_path / "train.txt"
    annot_file.write_text("img1.jpg 0\n\nimg2.jpg 4\n", encoding="utf-8")

    records = load_annotations(annot_file, img_dir)

    assert len(records) == 2
    assert records[0].image_path == (img_dir / "img1.jpg").resolve()
    assert records[0].label == 0
    assert records[1].label == 4


def test_load_annotations_missing_annotation_file(tmp_path: Path) -> None:
    """Test exception when annotation file does not exist."""
    annot_file = tmp_path / "non_existent.txt"
    img_dir = tmp_path / "images"
    img_dir.mkdir()

    with pytest.raises(AnnotationFileNotFoundError):
        load_annotations(annot_file, img_dir)


def test_load_annotations_missing_image_directory(tmp_path: Path) -> None:
    """Test exception when image directory does not exist or is not a directory."""
    annot_file = tmp_path / "train.txt"
    annot_file.touch()
    
    non_existent_dir = tmp_path / "no_dir"
    with pytest.raises(InvalidAnnotationError) as exc_info:
        load_annotations(annot_file, non_existent_dir)
    assert "Image directory not found" in str(exc_info.value)

    # Передаем файл вместо директории
    file_as_dir = tmp_path / "some_file.txt"
    file_as_dir.touch()
    with pytest.raises(InvalidAnnotationError) as exc_info:
        load_annotations(annot_file, file_as_dir)
    assert "not a directory" in str(exc_info.value)


def test_load_annotations_invalid_format(tmp_path: Path) -> None:
    """Test exception when a line has too few or too many columns."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    
    annot_file = tmp_path / "invalid_format.txt"
    annot_file.write_text("img1.jpg 0 extra_part\n", encoding="utf-8")

    with pytest.raises(InvalidAnnotationError) as exc_info:
        load_annotations(annot_file, img_dir)
    assert "Invalid annotation format" in str(exc_info.value)


def test_load_annotations_invalid_label_value(tmp_path: Path) -> None:
    """Test exception when class label is not an integer or is out of range [0-4]."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    (img_dir / "img1.jpg").touch()

    # Нечисловая метка
    annot_file = tmp_path / "bad_label.txt"
    annot_file.write_text("img1.jpg abc\n", encoding="utf-8")
    with pytest.raises(InvalidAnnotationError):
        load_annotations(annot_file, img_dir)

    # Метка вне диапазона
    annot_file_out_of_range = tmp_path / "out_of_range.txt"
    annot_file_out_of_range.write_text("img1.jpg 5\n", encoding="utf-8")
    with pytest.raises(InvalidAnnotationError):
        load_annotations(annot_file_out_of_range, img_dir)


def test_load_annotations_missing_referenced_image(tmp_path: Path) -> None:
    """Test exception when the referenced image file does not exist on disk."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()

    annot_file = tmp_path / "missing_img.txt"
    annot_file.write_text("ghost.jpg 1\n", encoding="utf-8")

    with pytest.raises(InvalidAnnotationError) as exc_info:
        load_annotations(annot_file, img_dir)
    assert "Image not found" in str(exc_info.value)


def test_load_annotations_empty_file(tmp_path: Path) -> None:
    """Test exception when the annotation file contains no valid rows."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()

    annot_file = tmp_path / "empty.txt"
    annot_file.write_text("\n  \n", encoding="utf-8")

    with pytest.raises(EmptyDatasetError):
        load_annotations(annot_file, img_dir)