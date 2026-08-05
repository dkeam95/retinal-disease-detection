"""Unit tests for the lesion detection model factory and config loading."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from common.config.types import DetectionModelConfig
from detection.detector_model import build_lesion_detector


def test_build_lesion_detector_default() -> None:
    """Verify that build_lesion_detector creates a valid PyTorch detector module."""
    model = build_lesion_detector(num_classes=4, pretrained=False)
    assert isinstance(model, nn.Module)


def test_build_lesion_detector_with_config() -> None:
    """Verify build_lesion_detector initializes from DetectionModelConfig."""
    config = DetectionModelConfig(
        architecture="fasterrcnn_resnet50_fpn",
        pretrained=False,
        num_classes=4,
        score_thresh=0.20,
        nms_thresh=0.35,
        min_size=1024,
        max_size=1024,
    )
    model = build_lesion_detector(config=config)
    assert isinstance(model, nn.Module)


def test_detector_forward_pass_dummy() -> None:
    """Verify detector accepts image tensor batch in train mode."""
    config = DetectionModelConfig(
        architecture="fasterrcnn_resnet50_fpn",
        pretrained=False,
        num_classes=4,
        min_size=1024,
        max_size=1024,
    )
    model = build_lesion_detector(config=config)
    model.train()

    dummy_image = torch.rand((3, 1024, 1024), dtype=torch.float32)
    dummy_target = {
        "boxes": torch.tensor([[100.0, 100.0, 300.0, 300.0]], dtype=torch.float32),
        "labels": torch.tensor([1], dtype=torch.int64),
    }

    loss_dict = model([dummy_image], [dummy_target])
    assert isinstance(loss_dict, dict)
    assert "loss_classifier" in loss_dict or "loss_rpn_box_reg" in loss_dict

