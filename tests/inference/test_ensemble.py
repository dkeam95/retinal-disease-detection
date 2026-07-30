"""Unit tests for EnsemblePredictor."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from common.config.types import (
    CheckpointConfig,
    DataLoaderConfig,
    DatasetConfig,
    EarlyStoppingConfig,
    ExperimentConfig,
    LoggingConfig,
    LossConfig,
    MetricsConfig,
    MixedPrecisionConfig,
    ModelConfig,
    OptimizerConfig,
    PreprocessingConfig,
    ProjectConfig,
    SchedulerConfig,
    TrainingConfig,
)
from inference.ensemble import EnsemblePredictor
from inference.predictor import RetinalPredictor
from training.model_factory import ModelFactory


@pytest.fixture
def dummy_config() -> ProjectConfig:
    return ProjectConfig(
        dataset=DatasetConfig(
            path="data/raw",
            annotation_file="train.txt",
            image_directory="train",
            num_classes=5,
        ),
        preprocessing=PreprocessingConfig(
            image_size=(224, 224),
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            horizontal_flip_prob=0.5,
            vertical_flip_prob=0.5,
            rotation_limit=180,
            rotation_prob=0.5,
            brightness_contrast_prob=0.4,
        ),
        dataloader=DataLoaderConfig(
            batch_size=4,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
            drop_last=False,
            persistent_workers=False,
        ),
        training=TrainingConfig(epochs=1, device="cpu"),
        model=ModelConfig(
            architecture="resnet50", pretrained=False, num_classes=5, dropout_rate=0.3
        ),
        loss=LossConfig(name="cross_entropy"),
        metrics=MetricsConfig(
            primary=("accuracy", "qwk"), secondary=("macro_f1",), per_class=True
        ),
        experiment=ExperimentConfig(name="test_exp", seed=42),
        optimizer=OptimizerConfig(name="adamw", lr=0.001, weight_decay=0.0001),
        scheduler=SchedulerConfig(name="cosine", eta_min=1e-6),
        checkpoint=CheckpointConfig(directory="checkpoints", monitor="val_loss"),
        early_stopping=EarlyStoppingConfig(enabled=True, patience=5, min_delta=0.001),
        mixed_precision=MixedPrecisionConfig(enabled=False, dtype="float16"),
        logging=LoggingConfig(log_every_n_steps=10),
    )


def test_ensemble_predictor(dummy_config):
    model1 = ModelFactory.build(
        architecture="resnet50", pretrained=False, num_classes=5
    )
    model2 = ModelFactory.build(
        architecture="densenet121", pretrained=False, num_classes=5
    )

    p1 = RetinalPredictor(model=model1, config=dummy_config, device="cpu")
    p2 = RetinalPredictor(model=model2, config=dummy_config, device="cpu")

    ensemble = EnsemblePredictor(predictors=[p1, p2], weights=[0.6, 0.4])

    dummy_rgb = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    result = ensemble.predict(dummy_rgb)

    assert 0 <= result.grade_id <= 4
    assert len(result.probabilities) == 5
    assert pytest.approx(sum(result.probabilities), 1e-4) == 1.0
