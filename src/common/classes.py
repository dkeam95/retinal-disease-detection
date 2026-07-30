"""
Diabetic Retinopathy class definitions.

This module contains the canonical class identifiers used
throughout the project.
"""

from __future__ import annotations  # Enables modern type hints (Python 3.7+)

from enum import IntEnum  # Enum class where members are also instances of int


class DRClass(IntEnum):
    """Diabetic Retinopathy severity classes."""

    # Canonical medical stage mappings (0 to 4 ICDR scale)
    NO_DR = 0  # Stage 0: Healthy retina without clinical signs
    MILD_NPDR = (
        1  # Stage 1: Mild Nonproliferative Diabetic Retinopathy (microaneurysms only)
    )
    MODERATE_NPDR = 2  # Stage 2: Moderate Nonproliferative Diabetic Retinopathy
    SEVERE_NPDR = 3  # Stage 3: Severe Nonproliferative Diabetic Retinopathy
    PROLIFERATIVE_DR = 4  # Stage 4: Proliferative Diabetic Retinopathy (advanced stage)


# Human-readable label mapping corresponding index-by-index to DRClass values
CLASS_NAMES: tuple[str, ...] = (
    "No Diabetic Retinopathy",
    "Mild Nonproliferative DR",
    "Moderate Nonproliferative DR",
    "Severe Nonproliferative DR",
    "Proliferative DR",
)
