"""
Annotation parser for retinal datasets.

This module is responsible for reading dataset annotation files
and converting them into AnnotationRecord objects.
"""

from __future__ import annotations

from pathlib import Path

from dataset.exceptions import (
    AnnotationFileNotFoundError,
    InvalidAnnotationError,
    EmptyDatasetError,
)

from dataset.types import AnnotationRecord


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

    if not annotation_file.exists():
        raise AnnotationFileNotFoundError(
            f"Annotation file not found: {annotation_file}"
        )

    annotations: list[AnnotationRecord] = []

    with annotation_file.open("r", encoding="utf-8") as file:

        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) != 2:
                raise InvalidAnnotationError(
                    f"Invalid annotation format "
                    f"at line {line_number}: {line}"
                )

            filename, label_text = parts

            try:
                label = int(label_text)

            except ValueError as error:
                raise InvalidAnnotationError(
                    f"Invalid class label "
                    f"at line {line_number}: {label_text}"
                ) from error

            if label < 0:
                raise InvalidAnnotationError(
                    f"Negative class label "
                    f"at line {line_number}: {label}"
                )

            image_path = image_directory / filename

            if not image_path.exists():
                raise InvalidAnnotationError(
                    f"Image not found: {image_path}"
                )

            annotations.append(
                AnnotationRecord(
                    image_path=image_path,
                    label=label,
                )
            )

    if not annotations:
        raise EmptyDatasetError(
            "The annotation file contains no samples."
        )

    return annotations