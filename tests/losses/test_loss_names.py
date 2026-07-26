"""
Unit tests for loss name enumerations.

This module contains unit tests verifying that all supported `LossName` enum members
correctly map to their expected canonical string representations.
"""

from __future__ import annotations  # Enables modern type hints (Python 3.7+)

from losses.loss_names import (
    LossName,  # Enum defining canonical string names for loss functions
)


def test_loss_names() -> None:
    """
    Verify that all supported loss name enum values match their expected string values.
    """

    # Confirm CROSS_ENTROPY maps to 'cross_entropy' string
    assert (
        LossName.CROSS_ENTROPY
        == "cross_entropy"
    )

    # Confirm WEIGHTED_CROSS_ENTROPY maps to 'weighted_cross_entropy' string
    assert (
        LossName.WEIGHTED_CROSS_ENTROPY
        == "weighted_cross_entropy"
    )

    # Confirm FOCAL maps to 'focal' string
    assert (
        LossName.FOCAL
        == "focal"
    )

    # Confirm CLASS_BALANCED_FOCAL maps to 'class_balanced_focal' string
    assert (
        LossName.CLASS_BALANCED_FOCAL
        == "class_balanced_focal"
    )