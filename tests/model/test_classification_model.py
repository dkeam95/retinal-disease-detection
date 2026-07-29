"""
Unit tests for the end-to-end classification model wrapper.

This module contains unit tests verifying that `ClassificationModel` correctly coordinates
the forward pass between the feature extraction backbone and the classification head,
returning outputs with expected shapes for logits and intermediate features.
"""

from __future__ import annotations  # Enables modern type hints (Python 3.7+)

import torch  # PyTorch tensor library
from torch import nn  # Base module class for neural networks

from model.classification_model import (
    ClassificationModel,  # Wrapper model combining backbone and classifier
)


class DummyBackbone(nn.Module):
    """
    Mock feature extraction backbone producing fixed-dimension outputs for testing.
    """

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Simulate feature extraction by returning random tensors of shape (batch_size, 1280).
        """
        return torch.randn(
            x.shape[0],
            1280,
        )


class DummyClassifier(nn.Module):
    """
    Mock classification head mapping 1280-dim feature vectors to class logits.
    """

    def __init__(self) -> None:
        super().__init__()

        self.fc = nn.Linear(
            1280,
            5,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute output logits for the input feature vectors.
        """
        return self.fc(x)


def test_classification_model() -> None:
    """
    Verify that `ClassificationModel` outputs logits and features with expected tensor shapes.
    """

    # Instantiate composite classification model using dummy modules
    model = ClassificationModel(
        backbone=DummyBackbone(),
        classifier=DummyClassifier(),
    )

    # Create dummy input batch (batch_size=2, channels=3, height=224, width=224)
    x = torch.randn(
        2,
        3,
        224,
        224,
    )

    # Perform forward pass through classification model
    output = model(
        x,
    )

    # Verify output logits shape corresponds to (batch_size, num_classes)
    assert output.logits.shape == (
        2,
        5,
    )

    # Verify extracted features shape corresponds to (batch_size, feature_dim)
    assert output.features.shape == (
        2,
        1280,
    )
