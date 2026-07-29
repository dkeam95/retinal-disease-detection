"""
Unit tests for model exceptions and error hierarchy.

This module contains unit tests verifying that custom model-related exceptions can be
properly raised and caught, ensuring consistent error handling across the model package.
"""

from __future__ import annotations  # Enables modern type hints (Python 3.7+)

import pytest  # Testing framework for asserting exceptions and test execution

from model.exceptions import (  # Custom exception hierarchy for model-related errors
    ModelError,
    ModelInitializationError,
    UnknownModelArchitectureError,
)


def test_model_error() -> None:
    """
    Verify that the base ModelError exception can be raised and caught.
    """

    # Assert raising base ModelError is correctly caught
    with pytest.raises(
        ModelError,
    ):
        raise ModelError(
            "Model error.",
        )


def test_unknown_model_architecture_error() -> None:
    """
    Verify that UnknownModelArchitectureError can be raised and caught.
    """

    # Assert raising UnknownModelArchitectureError is correctly caught
    with pytest.raises(
        UnknownModelArchitectureError,
    ):
        raise UnknownModelArchitectureError(
            "Unknown model.",
        )


def test_model_initialization_error() -> None:
    """
    Verify that ModelInitializationError can be raised and caught.
    """

    # Assert raising ModelInitializationError is correctly caught
    with pytest.raises(
        ModelInitializationError,
    ):
        raise ModelInitializationError(
            "Initialization failed.",
        )
