"""
Annotation parser for retinal datasets.

This module is responsible for reading dataset annotation files
and converting them into AnnotationRecord objects.
"""

from __future__ import annotations  # Enables modern type hints (Python 3.7+)

from pathlib import Path  # Object-oriented filesystem path navigation

# Domain-specific dataset exceptions for missing files, parsing errors, or empty datasets
from dataset.exceptions import (
    AnnotationFileNotFoundError,
    EmptyDatasetError,
    InvalidAnnotationError,
)
from dataset.types import (
    AnnotationRecord,  # Dataclass structure representing a single sample record
)


def load_annotations(
    annotation_file: Path,
    image_directory: Path,
) -> list[AnnotationRecord]:
    """
    Load annotation records from a dataset annotation file.

    Parameters
    ----------
    annotation_file : Path
        Path to the annotation file.

    image_directory : Path
        Directory containing dataset images.

    Returns
    -------
    list[AnnotationRecord]
        Parsed annotation records.

    Raises
    ------
    AnnotationFileNotFoundError
        If the annotation file does not exist.

    InvalidAnnotationError
        If an annotation record has an invalid format or
        references a missing image.

    EmptyDatasetError
        If the annotation file contains no valid samples.
    """

    # Check if the requested annotation file exists before attempting to read
    if not annotation_file.exists():
        raise AnnotationFileNotFoundError(
            f"Annotation file not found: {annotation_file}"
        )

    # Ensure image directory exists
    if not image_directory.exists():
        raise InvalidAnnotationError(f"Image directory not found: {image_directory}")

    # Ensure provided path is actually a directory
    if not image_directory.is_dir():
        raise InvalidAnnotationError(
            f"Image directory is not a directory: {image_directory}"
        )

    # Empty list to store processed annotation records
    annotations: list[AnnotationRecord] = []

    # Safely open and read annotation file line-by-line using UTF-8 encoding
    with annotation_file.open("r", encoding="utf-8") as file:
        # Iterate over each line in the annotation file
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            # Skip empty lines or whitespace-only lines
            if not line:
                continue

            # Split line into components expecting exactly [filename, label]
            parts = line.split()

            # Check if the line contains exactly 2 components: the filename and the class label
            if len(parts) != 2:
                raise InvalidAnnotationError(
                    f"Invalid annotation format at line {line_number}: {line}"
                )

            filename, label_text = parts

            # Parse string class label into integer
            try:
                label = int(label_text)

            except ValueError as error:
                raise InvalidAnnotationError(
                    f"Invalid class label at line {line_number}: {label_text}"
                ) from error

            # Validate class label range
            # DDR dataset:
            # label 5 = poor-quality fundus image.
            # These images are excluded from DR grading.
            if label == 5:
                continue

            if label not in range(5):
                raise InvalidAnnotationError(
                    f"Invalid class label at line {line_number}: {label}"
                )

            # Construct absolute/resolved target path for the image file
            image_path = (image_directory / filename).resolve()

            # Ensure the referenced image file exists on the filesystem
            if not image_path.exists():
                raise InvalidAnnotationError(f"Image not found: {image_path}")

            # Record validated sample entry
            annotations.append(
                AnnotationRecord(
                    image_path=image_path,
                    label=label,
                )
            )

    # Ensure dataset contains at least one valid sample entry
    if not annotations:
        raise EmptyDatasetError("The annotation file contains no samples.")

    return annotations
