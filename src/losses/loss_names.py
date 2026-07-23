"""Supported loss function names."""

from __future__ import annotations

from enum import StrEnum


class LossName(StrEnum):
    """Supported loss functions."""

    CROSS_ENTROPY = "cross_entropy"
    WEIGHTED_CROSS_ENTROPY = "weighted_cross_entropy"
    FOCAL = "focal"
    CLASS_BALANCED_FOCAL = "class_balanced_focal"
    
    