"""Annotation parser for retinal datasets.

This module is responsible for reading dataset annotation files
and converting them into AnnotationRecord objects.
"""

from __future__ import annotations

from pathlib import Path

from dataset.exceptions import (
    AnnotationFileNotFoundError,
    InvalidAnnotationError
)

from dataset.types import AnnotationRecord


def load_annotations(annotation_file: Path, image_directory: Path) -> list[AnnotationRecord]:
    """Load annotation records from a dataset annotation file.
    
    Args:
        annotation_file: Path to the annotation file.
        image_directory: Directory containing dataset images.
        
    Returns:
        list[AnnotationRecord]: Parsed annotation records.

    Raises:
        AnnotationFileNotFoundError: If the annotation file  does not exist.
        InvalidAnnotationError: If an annotation record has invalid format.
    """

    if not annotation_file.exists():
        raise AnnotationFileNotFoundError(
            f"Annotation file not found: {annotation_file}"
        )

    annotations: list[AnnotationRecord] = []

    with annotation_file.open(mode="r", encoding="utf-8") as file:

        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) != 2:
                raise InvalidAnnotationError(
                    f"Invalid annotation format at line "
                    f"{line_number}: {line}"
                )
            
            filename, label_text = parts

            try:
                label = int(label_text)
            
            except ValueError as error:
                raise InvalidAnnotationError(
                    f"Invalid class label at line "
                    f"{line_number}: {label_text}"
                ) from error

            annotations.append(
                AnnotationRecord(
                    image_path=image_directory / filename,
                    label=label
                )
            )

    return annotations