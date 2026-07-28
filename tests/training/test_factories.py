"""
Tests for training factories.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    OneCycleLR,
    StepLR,
)

from training.exceptions import (
    LossFactoryError,
    ModelFactoryError,
    OptimizerFactoryError,
    SchedulerFactoryError,
)

from training.loss_factory import LossFactory
from training.model_factory import ModelFactory
from training.optimizer_factory import OptimizerFactory
from training.scheduler_factory import SchedulerFactory


@pytest.fixture
def dummy_model() -> nn.Module:
    """
    Simple model used for optimizer and scheduler tests.
    """

    return nn.Linear(10, 5)


# ==========================================================
# LossFactory
# ==========================================================


def test_build_cross_entropy() -> None:
    """
    Test that CrossEntropyLoss is created.
    """

    criterion = LossFactory.build(
        name="cross_entropy",
    )

    assert isinstance(
        criterion,
        nn.CrossEntropyLoss,
    )


def test_build_cross_entropy_with_label_smoothing() -> None:
    """
    Test label smoothing parameter.
    """

    criterion = LossFactory.build(
        name="cross_entropy",
        label_smoothing=0.1,
    )

    assert isinstance(
        criterion,
        nn.CrossEntropyLoss,
    )

    assert criterion.label_smoothing == pytest.approx(
        0.1,
    )


def test_build_cross_entropy_with_class_weights() -> None:
    """
    Test class weights.
    """

    weights = torch.tensor(
        [
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
        ],
        dtype=torch.float32,
    )

    criterion = LossFactory.build(
        name="cross_entropy",
        weight=weights,
    )

    assert isinstance(
        criterion,
        nn.CrossEntropyLoss,
    )

    assert criterion.weight is not None

    assert torch.equal(
        criterion.weight,
        weights,
    )


def test_unknown_loss() -> None:
    """
    Unknown loss should raise an exception.
    """

    with pytest.raises(
        LossFactoryError,
    ):
        LossFactory.build(
            name="unknown_loss",
        )


# ==========================================================
# OptimizerFactory
# ==========================================================


def test_build_adam(
    dummy_model: nn.Module,
) -> None:
    """
    Test Adam optimizer creation.
    """

    optimizer = OptimizerFactory.build(
        name="adam",
        parameters=dummy_model.parameters(),
        learning_rate=1e-3,
    )

    assert isinstance(
        optimizer,
        Adam,
    )


def test_build_adamw(
    dummy_model: nn.Module,
) -> None:
    """
    Test AdamW optimizer creation.
    """

    optimizer = OptimizerFactory.build(
        name="adamw",
        parameters=dummy_model.parameters(),
        learning_rate=1e-3,
        weight_decay=1e-4,
    )

    assert isinstance(
        optimizer,
        AdamW,
    )

    assert optimizer.defaults["weight_decay"] == pytest.approx(
        1e-4,
    )


def test_build_sgd(
    dummy_model: nn.Module,
) -> None:
    """
    Test SGD optimizer creation.
    """

    optimizer = OptimizerFactory.build(
        name="sgd",
        parameters=dummy_model.parameters(),
        learning_rate=1e-2,
        momentum=0.9,
    )

    assert isinstance(
        optimizer,
        SGD,
    )

    assert optimizer.defaults["momentum"] == pytest.approx(
        0.9,
    )


def test_unknown_optimizer(
    dummy_model: nn.Module,
) -> None:
    """
    Unknown optimizer should raise an exception.
    """

    with pytest.raises(
        OptimizerFactoryError,
    ):
        OptimizerFactory.build(
            name="unknown_optimizer",
            parameters=dummy_model.parameters(),
            learning_rate=1e-3,
        )


# ==========================================================
# SchedulerFactory
# ==========================================================


def test_build_cosine_scheduler(
    dummy_model: nn.Module,
) -> None:
    """
    Test CosineAnnealingLR creation.
    """

    optimizer = AdamW(
        dummy_model.parameters(),
        lr=1e-3,
    )

    scheduler = SchedulerFactory.build(
        name="cosine",
        optimizer=optimizer,
        epochs=100,
    )

    assert isinstance(
        scheduler,
        CosineAnnealingLR,
    )


def test_build_step_scheduler(
    dummy_model: nn.Module,
) -> None:
    """
    Test StepLR creation.
    """

    optimizer = AdamW(
        dummy_model.parameters(),
        lr=1e-3,
    )

    scheduler = SchedulerFactory.build(
        name="step",
        optimizer=optimizer,
        epochs=100,
        step_size=20,
        gamma=0.5,
    )

    assert isinstance(
        scheduler,
        StepLR,
    )

    assert scheduler.step_size == 20
    assert scheduler.gamma == pytest.approx(
        0.5,
    )


def test_build_onecycle_scheduler(
    dummy_model: nn.Module,
) -> None:
    """
    Test OneCycleLR creation.
    """

    optimizer = AdamW(
        dummy_model.parameters(),
        lr=1e-3,
    )

    scheduler = SchedulerFactory.build(
        name="onecycle",
        optimizer=optimizer,
        epochs=10,
        steps_per_epoch=100,
        max_learning_rate=1e-3,
    )

    assert isinstance(
        scheduler,
        OneCycleLR,
    )


def test_onecycle_requires_steps_per_epoch(
    dummy_model: nn.Module,
) -> None:
    """
    OneCycleLR requires steps_per_epoch.
    """

    optimizer = AdamW(
        dummy_model.parameters(),
        lr=1e-3,
    )

    with pytest.raises(
        SchedulerFactoryError,
    ):
        SchedulerFactory.build(
            name="onecycle",
            optimizer=optimizer,
            epochs=10,
            max_learning_rate=1e-3,
        )


def test_onecycle_requires_max_learning_rate(
    dummy_model: nn.Module,
) -> None:
    """
    OneCycleLR requires max_learning_rate.
    """

    optimizer = AdamW(
        dummy_model.parameters(),
        lr=1e-3,
    )

    with pytest.raises(
        SchedulerFactoryError,
    ):
        SchedulerFactory.build(
            name="onecycle",
            optimizer=optimizer,
            epochs=10,
            steps_per_epoch=100,
        )


def test_unknown_scheduler(
    dummy_model: nn.Module,
) -> None:
    """
    Unknown scheduler should raise an exception.
    """

    optimizer = AdamW(
        dummy_model.parameters(),
        lr=1e-3,
    )

    with pytest.raises(
        SchedulerFactoryError,
    ):
        SchedulerFactory.build(
            name="unknown_scheduler",
            optimizer=optimizer,
            epochs=100,
        )


# ==========================================================
# ModelFactory
# ==========================================================


def test_build_efficientnet_b0() -> None:
    """
    Test EfficientNet-B0 creation.
    """

    model = ModelFactory.build(
        architecture="efficientnet_b0",
        pretrained=False,
        num_classes=5,
    )

    assert isinstance(
        model,
        nn.Module,
    )


def test_classifier_output_features() -> None:
    """
    Test classifier output dimension.
    """

    model = ModelFactory.build(
        architecture="efficientnet_b0",
        pretrained=False,
        num_classes=5,
    )

    classifier = model.classifier[-1]

    assert isinstance(
        classifier,
        nn.Linear,
    )

    assert classifier.out_features == 5


def test_pretrained_model_creation() -> None:
    """
    Test pretrained EfficientNet-B0 creation.
    """

    model = ModelFactory.build(
        architecture="efficientnet_b0",
        pretrained=True,
        num_classes=5,
    )

    assert isinstance(
        model,
        nn.Module,
    )


def test_unknown_model_architecture() -> None:
    """
    Unknown architecture should raise an exception.
    """

    with pytest.raises(
        ModelFactoryError,
    ):
        ModelFactory.build(
            architecture="unknown_model",
            pretrained=False,
            num_classes=5,
        )


def test_count_parameters() -> None:
    """
    Test parameter counting.
    """

    model = ModelFactory.build(
        architecture="efficientnet_b0",
        pretrained=False,
        num_classes=5,
    )

    total, trainable = ModelFactory.count_parameters(
        model,
    )

    assert total > 0
    assert trainable > 0
    assert total >= trainable


def test_classifier_initialization() -> None:
    """
    Test classifier initialization.
    """

    model = ModelFactory.build(
        architecture="efficientnet_b0",
        pretrained=False,
        num_classes=5,
    )

    classifier = model.classifier[-1]

    assert classifier.bias is not None

    assert torch.allclose(
        classifier.bias,
        torch.zeros_like(classifier.bias),
    )