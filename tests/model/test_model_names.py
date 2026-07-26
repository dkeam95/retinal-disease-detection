"""
Unit tests for model architecture enumerations.

This module contains unit tests verifying that `ModelArchitecture` enum values correctly
behave as standard strings and support reverse lookup from string representations.
"""

from __future__ import annotations  # Enables modern type hints (Python 3.7+)

from model.model_names import (
    ModelArchitecture,  # Enum defining supported model architecture names
)


def test_model_architecture_is_string() -> None:
    """
    Verify that ModelArchitecture enum values equal their expected string representations.
    """

    # Confirm EFFICIENTNET_B0 enum value equals string 'efficientnet_b0'
    assert (
        ModelArchitecture.EFFICIENTNET_B0
        == "efficientnet_b0"
    )


def test_model_lookup() -> None:
    """
    Verify that ModelArchitectureenum instance can be retrieved from its string value.
    """

    # Convert string 'efficientnet_b0' into corresponding ModelArchitecture enum instance
    architecture = ModelArchitecture(
        "efficientnet_b0"
    )

    # Assert string lookup yields correct EFFICIENTNET_B0 enum member
    assert (
        architecture
        is ModelArchitecture.EFFICIENTNET_B0
    )