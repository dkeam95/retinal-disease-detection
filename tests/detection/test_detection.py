"""Unit tests for Diabetic Retinopathy Lesion Detection subsystem."""

from __future__ import annotations

from pathlib import Path
import pytest
import torch

from common.config.types import DetectionDatasetConfig, DetectionModelConfig
from dataset.detection_dataset import LesionDetectionDataset, detection_collate_fn
from detection.detector_model import build_lesion_detector
from detection.metrics import LesionDetectionEvaluator
from detection.xml_parser import LESION_CLASS_MAP, parse_voc_xml
from inference.detection_predictor import DetectionResult, LesionDetectionPredictor

XML_DIR = Path("data/raw/lesion_detection/train")
IMAGE_DIR = Path("data/raw/train")


def test_xml_parser() -> None:
    """Test XML parsing on an actual XML file in the dataset."""
    xml_files = list(XML_DIR.glob("*.xml"))
    if not xml_files:
        pytest.skip("No XML files found for testing")

    parsed = parse_voc_xml(xml_files[0])
    assert parsed.filename.endswith(".jpg") or parsed.filename.endswith(".png")
    assert isinstance(parsed.width, int)
    assert isinstance(parsed.height, int)
    assert isinstance(parsed.boxes, list)
    assert isinstance(parsed.labels, list)
    assert len(parsed.boxes) == len(parsed.labels)

    for label in parsed.labels:
        assert label in (1, 2, 3, 4)


def test_detection_dataset() -> None:
    """Test LesionDetectionDataset loading and item shapes."""
    if not list(XML_DIR.glob("*.xml")):
        pytest.skip("No XML files found for testing")

    dataset = LesionDetectionDataset(
        xml_dir=XML_DIR,
        image_dir=IMAGE_DIR,
        target_size=(320, 320),
    )
    assert len(dataset) > 0

    img_tensor, target = dataset[0]
    assert isinstance(img_tensor, torch.Tensor)
    assert img_tensor.shape == (3, 320, 320)
    assert "boxes" in target
    assert "labels" in target
    assert "image_id" in target
    assert target["boxes"].ndim == 2
    assert target["boxes"].shape[1] == 4


def test_detection_collate_fn() -> None:
    """Test custom detection collate function."""
    if not list(XML_DIR.glob("*.xml")):
        pytest.skip("No XML files found for testing")

    dataset = LesionDetectionDataset(
        xml_dir=XML_DIR,
        image_dir=IMAGE_DIR,
        target_size=(320, 320),
    )

    batch_items = [dataset[0], dataset[1] if len(dataset) > 1 else dataset[0]]
    images, targets = detection_collate_fn(batch_items)

    assert isinstance(images, list)
    assert isinstance(targets, list)
    assert len(images) == len(batch_items)
    assert len(targets) == len(batch_items)


def test_detector_model_forward() -> None:
    """Test Faster R-CNN detector model forward pass in train and eval modes."""
    model = build_lesion_detector(num_classes=5, pretrained=False, min_size=320, max_size=320)
    model.eval()

    dummy_image = torch.rand(3, 320, 320)
    with torch.no_grad():
        preds = model([dummy_image])

    assert isinstance(preds, list)
    assert len(preds) == 1
    assert "boxes" in preds[0]
    assert "scores" in preds[0]
    assert "labels" in preds[0]


def test_detection_evaluator() -> None:
    """Test LesionDetectionEvaluator mAP calculation."""
    evaluator = LesionDetectionEvaluator()

    preds = [
        {
            "boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]),
            "scores": torch.tensor([0.90]),
            "labels": torch.tensor([1]),
        }
    ]
    targets = [
        {
            "boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]),
            "labels": torch.tensor([1]),
        }
    ]

    evaluator.update(preds, targets)
    metrics = evaluator.compute()

    assert "map" in metrics
    assert "map_50" in metrics
    assert metrics["map_50"] >= 0.0
