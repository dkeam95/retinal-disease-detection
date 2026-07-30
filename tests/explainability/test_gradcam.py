"""Unit tests for Grad-CAM explainability module."""

from __future__ import annotations

import numpy as np
import torch

from explainability.gradcam import GradCAM, overlay_heatmap
from training.model_factory import ModelFactory


def test_gradcam_generation(tmp_path):
    model = ModelFactory.build(
        architecture="efficientnet_b0", pretrained=False, num_classes=5
    )
    gradcam = GradCAM(model=model)

    # Input tensor (1, 3, 224, 224)
    input_tensor = torch.randn(1, 3, 224, 224, requires_grad=True)

    heatmap = gradcam.generate(input_tensor=input_tensor, target_class=0)

    assert heatmap is not None
    assert heatmap.shape == (224, 224)
    assert 0.0 <= heatmap.min() <= heatmap.max() <= 1.0

    # Test overlay helper
    dummy_rgb = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    blended = overlay_heatmap(dummy_rgb, heatmap, alpha=0.5)

    assert blended.shape == (224, 224, 3)
    assert blended.dtype == np.uint8

    # Test save visualization
    save_file = tmp_path / "gradcam.png"
    out_path = gradcam.save_visualization(
        input_tensor=input_tensor,
        original_rgb=dummy_rgb,
        save_path=save_file,
        target_class=0,
    )
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_save_experiment_visualization(tmp_path):
    model = ModelFactory.build(
        architecture="convnext_tiny", pretrained=False, num_classes=5
    )
    gradcam = GradCAM(model=model)
    input_tensor = torch.randn(1, 3, 224, 224, requires_grad=True)
    dummy_rgb = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

    out_path = gradcam.save_experiment_visualization(
        input_tensor=input_tensor,
        original_rgb=dummy_rgb,
        image_id="20170413102628830.jpg",
        experiment_or_model="exp_05_convnext_tiny",
        output_dir=tmp_path,
        true_label=3,
        predicted_label=3,
        confidence=0.9412,
    )

    assert out_path.exists()
    assert "exp_05_convnext_tiny" in str(out_path.parent)
    assert (
        out_path.name == "exp_05_convnext_tiny_20170413102628830_gt3_pred3_conf0.94.png"
    )
