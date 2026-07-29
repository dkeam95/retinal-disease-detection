"""Supported model architecture names.

This module is responsible for providing supported model architecture names.
"""

from __future__ import annotations  # Enables modern type hints (Python 3.7+)

from enum import StrEnum  # String enum base class (Python 3.11+)


class ModelArchitecture(StrEnum):
    """Supported model architecture."""

    EFFICIENTNET_B0 = "efficientnet_b0"
    RESNET50 = "resnet50"
    DENSENET121 = "densenet121"
    MOBILENET_V3_LARGE = "mobilenet_v3_large"
    CONVNEXT_TINY = "convnext_tiny"
    VIT_B_16 = "vit_b_16"
    SWIN_T = "swin_t"
