"""Faster R-CNN Model Factory for Diabetic Retinopathy Lesion Detection.

Implements Two-Stage Faster R-CNN with ResNet50-FPN backbone initialized from COCO pre-trained detection weights.
"""

from __future__ import annotations

import torch
from torch import nn
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN_ResNet50_FPN_Weights

from common.config.types import DetectionModelConfig


def build_lesion_detector(
    config: DetectionModelConfig | None = None,
    num_classes: int = 5,
    pretrained: bool = True,
    score_thresh: float = 0.25,
    nms_thresh: float = 0.45,
    min_size: int = 1536,
    max_size: int = 2048,
) -> FasterRCNN:
    """Build a Two-Stage Faster R-CNN (ResNet50-FPN) detector for lesion detection.

    Args:
        config: Optional DetectionModelConfig dataclass.
        num_classes: Number of detection classes (default: 5 [background + 4 lesions]).
        pretrained: Whether to initialize with COCO pretrained detection weights.
        score_thresh: Score threshold for bounding box predictions.
        nms_thresh: Non-Maximum Suppression IoU threshold.
        min_size: Minimum image size for detector input.
        max_size: Maximum image size for detector input.

    Returns:
        FasterRCNN PyTorch model ready for training or inference.
    """
    if config is not None:
        num_classes = config.num_classes
        pretrained = config.pretrained
        score_thresh = config.score_thresh
        nms_thresh = config.nms_thresh
        min_size = config.min_size
        max_size = config.max_size

    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None

    # Load pre-trained Faster R-CNN with ResNet-50-FPN backbone
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=weights,
        min_size=min_size,
        max_size=max_size,
        box_score_thresh=score_thresh,
        box_nms_thresh=nms_thresh,
    )

    # Customize Anchor Generator for 5 FPN feature levels (P2..P5 + pool)
    anchor_generator = AnchorGenerator(
        sizes=((4,), (8,), (16,), (32,), (64,)),
        aspect_ratios=((0.5, 1.0, 2.0),) * 5,
    )
    model.rpn.anchor_generator = anchor_generator

    # Replace the classification head for 5 output classes (background + 4 lesion classes)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model
