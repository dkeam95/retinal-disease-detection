import pytest

from losses.exceptions import (
    LossError,
    LossInitializationError,
    UnknownLossError,
)


def test_unknown_loss_error() -> None:
    """Verify UnknownLossError."""

    with pytest.raises(
        UnknownLossError,
    ):
        raise UnknownLossError(
            "Unknown loss."
        )


def test_loss_initialization_error() -> None:
    """Verify LossInitializationError."""

    with pytest.raises(
        LossInitializationError,
    ):
        raise LossInitializationError(
            "Initialization failed."
        )


def test_loss_error_inheritance() -> None:
    """Verify inheritance hierarchy."""

    assert issubclass(
        UnknownLossError,
        LossError,
    )

    assert issubclass(
        LossInitializationError,
        LossError,
    )