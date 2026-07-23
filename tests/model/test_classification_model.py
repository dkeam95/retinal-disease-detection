import torch
from torch import nn

from model.classification_model import ClassificationModel


class DummyBackbone(nn.Module):

    def forward(
        self,
        x,
    ):
        return torch.randn(
            x.shape[0],
            1280,
        )


class DummyClassifier(nn.Module):

    def __init__(self):
        super().__init__()

        self.fc = nn.Linear(
            1280,
            5,
        )

    def forward(
        self,
        x,
    ):
        return self.fc(x)


def test_classification_model():

    model = ClassificationModel(
        backbone=DummyBackbone(),
        classifier=DummyClassifier(),
    )

    x = torch.randn(
        2,
        3,
        224,
        224,
    )

    output = model(
        x,
    )

    assert output.logits.shape == (
        2,
        5,
    )

    assert output.features.shape == (
        2,
        1280,
    )