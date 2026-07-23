"""Common model type defenitions."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor


@dataclass(slots=True)
class ModelOutput:
    """
    Output of a classification model.

    Attributes:
        logits:
            Raw model predictions of shape (N, C).

        features:
            Feature vector extracted by the backbone.
    """

    logits: Tensor
    features: Tensor