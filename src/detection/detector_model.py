"""Model Factory for Diabetic Retinopathy Lesion Detection.

Supports:
1. Anchor-Free YOLOv8 / YOLOv11 detectors for micro-lesion detection.
2. Two-Stage Faster R-CNN with ResNet50-FPN backbone and calibrated anchor generators.
"""

from __future__ import annotations

import logging
import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import (
    FasterRCNN_ResNet50_FPN_Weights,
    FastRCNNPredictor,
)

from common.config.types import DetectionModelConfig

DETECTION_PIPELINE_VERSION = "anchor-free-yolo-v3"
logger = logging.getLogger(__name__)


def build_lesion_detector(
    config: DetectionModelConfig | None = None,
    num_classes: int = 4,
    pretrained: bool = True,
    score_thresh: float = 0.25,
    nms_thresh: float = 0.45,
    min_size: int = 1024,
    max_size: int = 1024,
) -> nn.Module:
    """Build a detector (YOLOv8/v11 Anchor-Free or Faster R-CNN) for lesion detection.

    Args:
        config: Optional DetectionModelConfig dataclass.
        num_classes: Number of detection classes (default: 4 [EX, HE, MA, SE] or 5 with bg).
        pretrained: Whether to initialize with pretrained detection weights.
        score_thresh: Score threshold for bounding box predictions.
        nms_thresh: Non-Maximum Suppression IoU threshold.
        min_size: Minimum image size for detector input.
        max_size: Maximum image size for detector input.

    Returns:
        PyTorch detector model ready for training or inference.
    """
    arch = "fasterrcnn_resnet50_fpn"
    if config is not None:
        arch = getattr(config, "architecture", "fasterrcnn_resnet50_fpn").lower()
        num_classes = config.num_classes
        pretrained = config.pretrained
        score_thresh = config.score_thresh
        nms_thresh = config.nms_thresh
        min_size = config.min_size
        max_size = config.max_size

    # 1. Build Anchor-Free YOLOv8 / YOLOv11 model if requested
    if "yolo" in arch:
        try:
            from ultralytics import YOLO
            logger.info(f"Building Anchor-Free detector: {arch}")
            yolo_model = YOLO(f"{arch}.pt" if pretrained else f"{arch}.yaml")
            return yolo_model.model
        except ImportError:
            logger.warning("ultralytics package not found. Falling back to PyTorch Anchor-Free Faster R-CNN.")

    # 2. Build Faster R-CNN with calibrated anchors and lower RPN IoU thresholds for micro-lesions
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=weights,
        min_size=min_size,
        max_size=max_size,
        box_score_thresh=score_thresh,
        box_nms_thresh=nms_thresh,
        rpn_fg_iou_thresh=0.45,  # Lowered from 0.70 to match 1024x1024 cropped micro-lesions
        rpn_bg_iou_thresh=0.20,
        rpn_post_nms_top_n_train=3000,
        rpn_post_nms_top_n_test=2000,
        box_detections_per_img=300,
    )

    # Calibrate Anchor Generator sizes for 1024x1024 tiles across 5 FPN levels (3 anchors per cell)
    anchor_generator = AnchorGenerator(
        sizes=((16,), (32,), (64,), (128,), (256,)),
        aspect_ratios=((0.5, 1.0, 2.0),) * 5,
    )
    model.rpn.anchor_generator = anchor_generator


    # Replace classification head for specified output classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

