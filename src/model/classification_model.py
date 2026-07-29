"""Classification model.

This model is a generic image classification model that is composed of a
backbone and a classifier.
"""

from __future__ import annotations  # Enables modern type hints (Python 3.7+)

from torch import Tensor, nn  # PyTorch tensor types and neural network modules

from model.types import (
    ModelOutput,  # Strongly typed container dataclass for model outputs
)


class ClassificationModel(nn.Module):
    """Generic image classification model."""

    def __init__(self, backbone: nn.Module, classifier: nn.Module) -> None:
        """Initialize composite classification architecture.

        Args:
            backbone:
                Feature extractor backbone network.

            classifier:
                Classification head module.
        """
        super().__init__()

        # Register feature extraction backbone and classification head submodules
        self.backbone = backbone
        self.classifier = classifier

    def forward(self, x: Tensor) -> ModelOutput:
        """Forward pass.

        Args:
            x:
                Input image batch tensor of shape (N, C, H, W).

        Returns:
            ModelOutput dataclass containing computed logits and feature embeddings.
        """

        # Extract deep feature embeddings from input images
        features = self.backbone(x)

        # Handle backbones returning multi-scale feature lists/tuples by selecting top layer
        if isinstance(features, (list, tuple)):
            features = features[-1]

        # Map feature vectors to class output logits
        logits = self.classifier(features)

        # Return structured container with both raw logits and feature embeddings
        return ModelOutput(
            logits=logits,
            features=features,
        )
