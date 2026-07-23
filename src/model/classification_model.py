"""Classification model."""

from __future__ import annotations

from torch import Tensor, nn

from model.types import ModelOutput


class ClassificationModel(nn.Module):
    """Generic image classification model."""

    def __init__(self, backbone: nn.Module, classifier: nn.Module) -> None:
        super().__init__()

        self.backbone = backbone
        self.classifier = classifier

    
    def forward(self, x: Tensor) -> ModelOutput:
        """
        Forward pass.

        Args:
            x:
                Input image batch.

        Returns:
            ModelOutput.
        """

        features = self.backbone(x)

        if isinstance(features, (list, tuple)):
            features = features[-1]

        logits = self.classifier(features)

        return ModelOutput(
            logits=logits,
            features=features,
        )
