"""Unit tests for GradCAMBatchAuditor."""

from __future__ import annotations

import numpy as np
import pytest

from explainability.batch_audit import GradCAMBatchAuditor
from inference.predictor import RetinalPredictor
from training.model_factory import ModelFactory


def test_gradcam_batch_auditor_export_gallery(tmp_path):
    model = ModelFactory.build(
        architecture="convnext_tiny", pretrained=False, num_classes=5
    )

    class DummyPredictor:
        def __init__(self, m):
            self.model = m
            self.device = "cpu"
            self.class_names = [
                "No DR",
                "Mild NPDR",
                "Moderate NPDR",
                "Severe NPDR",
                "Proliferative DR",
            ]

    predictor = DummyPredictor(model)
    auditor = GradCAMBatchAuditor(predictor=predictor, output_dir=tmp_path)

    samples = [
        {
            "image_id": "test1.jpg",
            "true_label": 0,
            "true_name": "No DR",
            "pred_label": 0,
            "pred_name": "No DR",
            "confidence": 0.95,
            "cam_path": str(tmp_path / "gradcam_audit" / "test1.png"),
        }
    ]

    out_file = auditor.export_html_gallery(samples, experiment_name="test_audit")
    assert out_file.exists()
    assert "gradcam_gallery.html" in str(out_file)
