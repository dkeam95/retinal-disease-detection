"""Public API for the losses module."""

from .factory import build_loss
from .registry import LOSS_REGISTRY

__all__ = [
    "build_loss",
    "LOSS_REGISTRY",
]