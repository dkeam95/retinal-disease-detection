"""
Model builders.

This module is responsible for building classification models
from pretrained backbones.
"""

from __future__ import annotations  # Enables modern type hints (Python 3.7+)

from torch import nn  # PyTorch neural network module base classes

from common.config.types import ModelConfig  # Strongly typed model configuration dataclass
from model.backbone import build_backbone  # Factory function for feature extraction backbones
from model.classification_model import ClassificationModel  # Wrapper assembling backbone and classifier
from model.classifier import build_classifier  # Factory function for classification heads


def build_timm_model(model_name: str, config: ModelConfig) -> nn.Module:
    """
    Build a classification model.

    Args:
        model_name:
            TIMM backbone name.

        config:
            Model configuration.

    Returns:
        Classification model.
    """

    # Construct feature extraction backbone without classification head
    backbone = build_backbone(
        model_name=model_name,
        config=config,
    )

    # Verify that the constructed backbone exposes feature dimension metadata
    if not hasattr(
        backbone,
        "num_features",
    ):
        raise AttributeError(
            f"Backbone '{model_name}' "
            "does not expose num_features."
        )

    # Construct classification head matching backbone's output feature dimension
    classifier = build_classifier(
        in_features=backbone.num_features,
        config=config,
    )

    # Assemble and return composite classification model
    return ClassificationModel(
        backbone=backbone,
        classifier=classifier,
    )