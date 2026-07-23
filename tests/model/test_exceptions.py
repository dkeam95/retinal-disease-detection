import pytest

from model.exceptions import (
    ModelError,
    ModelInitializationError,
    UnknownModelArchitectureError,
)


def test_model_error() -> None:
    """Verify base model exception."""

    with pytest.raises(
        ModelError,
    ):
        raise ModelError(
            "Model error.",
        )


def test_unknown_model_architecture_error() -> None:
    """Verify unknown architecture exception."""

    with pytest.raises(
        UnknownModelArchitectureError,
    ):
        raise UnknownModelArchitectureError(
            "Unknown model.",
        )


def test_model_initialization_error() -> None:
    """Verify initialization exception."""

    with pytest.raises(
        ModelInitializationError,
    ):
        raise ModelInitializationError(
            "Initialization failed.",
        )