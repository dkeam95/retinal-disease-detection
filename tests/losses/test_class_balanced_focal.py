"""
Unit tests for the Class-Balanced Focal Loss implementation.

This module contains unit tests verifying proper configuration validation,
including handling of unsupported reduction modes during loss initialization.
"""

from __future__ import annotations  # Enables modern type hints (Python 3.7+)

import pytest  # Testing framework for assertion checks
import torch  # PyTorch tensor library

from common.config.types import LossConfig  # Loss function configuration dataclass
from losses.class_balanced_focal import (
    build_class_balanced_focal_loss,  # Factory function for Class-Balanced Focal Loss
)
from losses.exceptions import (
    LossInitializationError,  # Exception raised on invalid loss initialization
)


def test_invalid_reduction() -> None:
    """
    Verify that an unsupported reduction mode raises LossInitializationError.
    """

    # Initialize dummy class weights tensor for 5 classes
    class_weights = torch.ones(
        5,
        dtype=torch.float32,
    )

    # Instantiate loss config with an unsupported reduction mode
    config = LossConfig(
        name="class_balanced_focal",
        gamma=2.0,
        reduction="invalid",
    )

    # Assert that invalid reduction type triggers LossInitializationError
    with pytest.raises(
        LossInitializationError,
    ):
        build_class_balanced_focal_loss(
            config,
            class_weights,
        )

    # Print success indicator after assertion passes
    print(
        "\nLossInitializationError successfully raised."
    )
