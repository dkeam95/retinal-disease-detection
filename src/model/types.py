"""Common model type definitions.

This module is responsible for providing common model type definitions.
"""

from __future__ import annotations  # Enables modern type hints (Python 3.7+)

from dataclasses import (
    dataclass,  # Decorator for creating lightweight container classes
)

from torch import Tensor  # PyTorch tensor object type annotation


@dataclass(slots=True)  # Optimize memory usage and access speed via slotted dataclass
class ModelOutput:
    """
    Output of a classification model.

    Attributes:
        logits:
            Raw model predictions of shape (N, C).

        features:
            Feature vector extracted by the backbone.
    """

    logits: Tensor  # Raw unnormalized prediction scores from classification head
    features: Tensor  # Intermediate feature embeddings extracted by backbone
