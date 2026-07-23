"""
Model builders.
"""

from __future__ import annotations

from torch import nn

from common.config.types import ModelConfig
from model.backbone import build_backbone
from model.classification_model import ClassificationModel
from model.classifier import build_classifier


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

    backbone = build_backbone(
        model_name=model_name,
        config=config,
    )

    if not hasattr(
        backbone,
        "num_features",
    ):
        raise AttributeError(
            f"Backbone '{model_name}' "
            "does not expose num_features."
        )

    classifier = build_classifier(
        in_features=backbone.num_features,
        config=config,
    )

    return ClassificationModel(
        backbone=backbone,
        classifier=classifier,
    )