"""
Unit tests for the loss factory module.

This module contains unit tests verifying that `build_loss` correctly instantiates standard
and weighted loss functions based on provided configurations and raises `UnknownLossError`
when provided with an unsupported loss name.
"""

from __future__ import annotations  # Enables modern type hints (Python 3.7+)

import pytest  # Testing framework for assertion checks
from torch import rand  # PyTorch function to generate random tensors

from common.config.types import LossConfig  # Loss function configuration dataclass
from losses.exceptions import UnknownLossError  # Exception raised for unsupported loss names
from losses.factory import build_loss  # Factory function for instantiating loss modules


def test_build_cross_entropy() -> None:
    """
    Verify that standard cross-entropy loss is successfully instantiated from configuration.
    """

    # Create valid configuration for default cross entropy
    config = LossConfig(
        name="cross_entropy",
        reduction="mean",
    )

    # Instantiate loss module via factory
    loss = build_loss(config)

    # Confirm loss instance was successfully created
    assert loss is not None


def test_unknown_loss() -> None:
    """
    Verify that providing an unsupported loss name to the factory raises UnknownLossError.
    """

    # Create configuration with an unsupported loss name
    config = LossConfig(
        name="invalid_loss",
        reduction="mean",
    )

    # Assert that factory raises UnknownLossError for invalid loss name
    with pytest.raises(
        UnknownLossError,
    ):
        build_loss(config)


def test_build_weighted_cross_entropy() -> None:
    """
    Verify that weighted cross-entropy loss is successfully instantiated with class weights.
    """

    # Create configuration for weighted cross entropy
    config = LossConfig(
        name="weighted_cross_entropy",
        reduction="mean",
    )

    # Generate random class weight vector for 5 classes
    weights = rand(5)

    # Instantiate weighted loss module passing weights vector
    loss = build_loss(
        config,
        weights,
    )

    # Confirm loss instance was successfully created
    assert loss is not None