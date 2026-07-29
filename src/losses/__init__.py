"""Public API for the losses module."""

# Imports from factory and registry to define explicit public interfaces
from .factory import build_loss  # Main factory function to construct loss modules
from .registry import (
    LOSS_REGISTRY,  # Dictionary registry mapping loss names to builders
)

# Explicitly declare public module exports to support clean wildcard imports
__all__ = [
    "build_loss",
    "LOSS_REGISTRY",
]
