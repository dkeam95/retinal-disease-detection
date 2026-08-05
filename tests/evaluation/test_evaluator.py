"""Unit tests for ModelEvaluator module."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset

from evaluation.evaluator import ModelEvaluator
from training.model_factory import ModelFactory


def test_model_evaluator(tmp_path):
    # Dummy model
    model = ModelFactory.build(architecture="efficientnet_b0", pretrained=False, num_classes=5)

    # Dummy dataset (10 samples of 3x224x224 and targets 0-4)
    images = torch.randn(10, 3, 224, 224)
    targets = torch.tensor([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
    dataset = TensorDataset(images, targets)
    loader = DataLoader(dataset, batch_size=2)

    evaluator = ModelEvaluator(model=model, device="cpu")

    report_path = tmp_path / "test_report.html"
    json_path = tmp_path / "test_metrics.json"

    results = evaluator.evaluate(
        dataloader=loader,
        save_report_path=report_path,
        save_json_path=json_path,
        experiment_name="Unit_Test_Eval",
    )

    assert "qwk" in results
    assert "accuracy" in results
    assert "macro_f1" in results
    assert "confusion_matrix" in results
    assert len(results["confusion_matrix"]) == 5
    assert report_path.exists()
    assert json_path.exists()
