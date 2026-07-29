"""Unit tests for RetinalPredictor module."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest
import torch

from common.config.loader import ConfigLoader
from inference.predictor import RetinalPredictor
from training.model_factory import ModelFactory

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_retinal_predictor():
    config_path = PROJECT_ROOT / "configs" / "config.yaml"
    config = ConfigLoader.load(config_path)

    model = ModelFactory.build(architecture="efficientnet_b0", pretrained=False, num_classes=5)
    predictor = RetinalPredictor(model=model, config=config, device="cpu")

    # Create dummy RGB numpy image (512, 512, 3)
    dummy_img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)

    result = predictor.predict(dummy_img)

    assert result.grade_id in range(5)
    assert isinstance(result.grade_name, str)
    assert 0.0 <= result.confidence <= 1.0
    assert len(result.probabilities) == 5
    assert pytest.approx(sum(result.probabilities), abs=1e-3) == 1.0
