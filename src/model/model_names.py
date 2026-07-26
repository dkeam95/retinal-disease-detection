"""Supported model architecture names.

This module is responsible for providing supported model architecture names.
"""

from __future__ import annotations  # Enables modern type hints (Python 3.7+)

from enum import StrEnum  # String enum base class (Python 3.11+)


class ModelArchitecture(StrEnum):
    """Supported model architecture."""

    EFFICIENTNET_B0 = "efficientnet_b0"  # Identifier for EfficientNet-B0 backbone architecture