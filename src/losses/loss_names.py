"""Supported loss function names."""

from __future__ import annotations              # Enables modern type hints (Python 3.7+)

from enum import StrEnum                        # String-based enumeration type


class LossName(StrEnum):
    """Supported loss functions."""

    CROSS_ENTROPY = "cross_entropy"                    # Standard cross-entropy loss function
    WEIGHTED_CROSS_ENTROPY = "weighted_cross_entropy"  # Class-weighted cross-entropy loss
    FOCAL = "focal"                                    # Standard focal loss for imbalanced classes
    CLASS_BALANCED_FOCAL = "class_balanced_focal"      # Class-balanced variation of focal loss