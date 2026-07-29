"""
Unit tests for loss exceptions and error hierarchy.

This module contains unit tests verifying that custom loss exceptions can be properly raised
and caught, and that the inheritance relationships between specific loss errors and the base
LossError class are correctly established.
"""

from __future__ import annotations  # Enables modern type hints (Python 3.7+)

import pytest  # Testing framework for assertion checks

from losses.exceptions import (
    LossError,  # Base exception class for loss module
    LossInitializationError,  # Exception raised on loss setup failure
    UnknownLossError,  # Exception raised when loss name is unrecognized
)


def test_unknown_loss_error() -> None:
    """
    Verify that UnknownLossError is raised and caught as expected.
    """

    # Assert that raising UnknownLossError is caught as expected
    with pytest.raises(
        UnknownLossError,
    ):
        raise UnknownLossError(
            "Unknown loss."
        )


def test_loss_initialization_error() -> None:
    """
    Verify that LossInitializationError is raised and caught as expected.
    """

    # Assert that raising LossInitializationError is caught as expected
    with pytest.raises(
        LossInitializationError,
    ):
        raise LossInitializationError(
            "Initialization failed."
        )


def test_loss_error_inheritance() -> None:
    """
    Verify that domain-specific loss exceptions correctly inherit from the base LossError class.
    """

    # Confirm UnknownLossError inherits from base LossError
    assert issubclass(
        UnknownLossError,
        LossError,
    )

    # Confirm LossInitializationError inherits from base LossError
    assert issubclass(
        LossInitializationError,
        LossError,
    )
