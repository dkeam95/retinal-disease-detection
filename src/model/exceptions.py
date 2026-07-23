"""Model-related exceptions."""

from __future__ import annotations


class ModelError(Exception):
    """Base exception for model module."""


class UnknownModelArchitectureError(ModelError):
    """Raised when an unknown model architecture is requested."""


class ModelInitializationError(ModelError):
    """Raised when model initialization fails."""