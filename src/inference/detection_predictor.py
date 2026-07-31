"""Inference Predictor and Visual Overlay Renderer for Lesion Bounding Box Detection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from common.config.types import DetectionModelConfig, ProjectConfig
from detection.detector_model import build_lesion_detector
from detection.xml_parser import LESION_ID_TO_NAME

# Color palette for medical bounding boxes (BGR format)
CLASS_COLOR_PALETTE_BGR: dict[int, tuple[int, int, int]] = {
    1: (0, 240, 255),  # Yellow -> EX (Hard Exudates)
    2: (0, 0, 255),    # Red -> HE (Hemorrhages)
    3: (255, 0, 255),  # Magenta -> MA (Microaneurysms)
    4: (255, 255, 0),  # Cyan -> SE (Soft Exudates)
}


@dataclass(frozen=True, slots=True)
class DetectionResult:
    """Dataclass storing detection predictions for a single image.

    Attributes:
        boxes: List of bounding boxes [xmin, ymin, xmax, ymax].
        scores: Confidence scores for each box.
        labels: Integer class IDs (1..4).
        class_names: Human-readable class names.
    """

    boxes: list[list[float]]
    scores: list[float]
    labels: list[int]
    class_names: list[str]


class LesionDetectionPredictor:
    """Predictor encapsulating Faster R-CNN detection inference and visual overlays."""

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        score_thresh: float = 0.25,
        target_size: tuple[int, int] = (1536, 2048),
    ) -> None:
        """Initialize LesionDetectionPredictor.

        Args:
            model: Trained Faster R-CNN model.
            device: Compute device (CUDA/CPU).
            score_thresh: Minimum prediction confidence score threshold.
            target_size: Model input resolution tuple (H, W).
        """
        self.model = model.to(device).eval()
        self.device = device
        self.score_thresh = score_thresh
        self.target_size = target_size

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        config: ProjectConfig,
        device: torch.device | None = None,
    ) -> LesionDetectionPredictor:
        """Instantiate predictor from saved checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file (.pt).
            config: ProjectConfig dataclass.
            device: Compute device. If None, auto-selects CUDA if available.

        Returns:
            Configured LesionDetectionPredictor instance.
        """
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        det_model_cfg = config.detection_model or DetectionModelConfig()
        model = build_lesion_detector(config=det_model_cfg)

        try:
            ckpt = torch.load(path, map_location=device, weights_only=False)
        except TypeError:
            ckpt = torch.load(path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])

        return cls(
            model=model,
            device=device,
            score_thresh=det_model_cfg.score_thresh,
            target_size=(det_model_cfg.min_size, det_model_cfg.max_size),
        )

    def predict(
        self,
        image_input: str | Path | np.ndarray,
    ) -> DetectionResult:
        """Perform object detection inference on an image.

        Args:
            image_input: Path to image file or RGB image numpy array (H, W, 3).

        Returns:
            DetectionResult dataclass containing predicted boxes, scores, and labels.
        """
        if isinstance(image_input, (str, Path)):
            bgr = cv2.imread(str(image_input))
            if bgr is None:
                raise ValueError(f"Failed to load image from: {image_input}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        else:
            rgb = image_input

        orig_h, orig_w = rgb.shape[:2]
        target_h, target_w = self.target_size

        # Letterbox aspect-ratio preserving scaling
        scale = min(float(target_w) / float(orig_w), float(target_h) / float(orig_h))
        new_w = int(round(orig_w * scale))
        new_h = int(round(orig_h * scale))
        pad_x = (target_w - new_w) / 2.0
        pad_y = (target_h - new_h) / 2.0

        resized_patch = cv2.resize(rgb, (new_w, new_h))
        img_resized = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        top_pad = int(round(pad_y))
        left_pad = int(round(pad_x))
        img_resized[top_pad : top_pad + new_h, left_pad : left_pad + new_w] = resized_patch

        img_tensor = (
            torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
        )
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std
        input_tensor = img_tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(input_tensor)[0]

        boxes = outputs["boxes"].cpu().numpy()
        scores = outputs["scores"].cpu().numpy()
        labels = outputs["labels"].cpu().numpy()

        res_boxes: list[list[float]] = []
        res_scores: list[float] = []
        res_labels: list[int] = []

        for box, score, label in zip(boxes, scores, labels, strict=False):
            if score < self.score_thresh:
                continue

            # Exact letterbox un-scaling to original coordinates without distortion or shift
            xmin = float((box[0] - left_pad) / scale)
            ymin = float((box[1] - top_pad) / scale)
            xmax = float((box[2] - left_pad) / scale)
            ymax = float((box[3] - top_pad) / scale)

            # Clip within image boundaries
            xmin = max(0.0, min(xmin, float(orig_w - 1)))
            ymin = max(0.0, min(ymin, float(orig_h - 1)))
            xmax = max(xmin + 1.0, min(xmax, float(orig_w)))
            ymax = max(ymin + 1.0, min(ymax, float(orig_h)))

            res_boxes.append([xmin, ymin, xmax, ymax])
            res_scores.append(float(score))
            res_labels.append(int(label))

        # Merge overlapping bounding boxes for the same lesion class to produce single exact boxes
        merged_boxes, merged_scores, merged_labels = self._merge_overlapping_boxes(
            res_boxes, res_scores, res_labels, iou_thresh=0.20
        )

        # Suppress false-positive bright exudate boxes (EX=1, SE=4) inside the anatomical Optic Disc
        from preprocessing.optic_disc import OpticDiscDetector
        od_detector = OpticDiscDetector(expansion_factor=1.25)
        od_circle = od_detector.detect(rgb)
        merged_boxes, merged_scores, merged_labels = od_detector.filter_boxes(
            merged_boxes, merged_scores, merged_labels, od_circle=od_circle, bright_class_ids={1, 4}
        )

        class_names = [LESION_ID_TO_NAME.get(int(l), f"class_{l}") for l in merged_labels]

        return DetectionResult(
            boxes=merged_boxes.tolist() if isinstance(merged_boxes, np.ndarray) else merged_boxes,
            scores=merged_scores.tolist() if isinstance(merged_scores, np.ndarray) else merged_scores,
            labels=merged_labels.tolist() if isinstance(merged_labels, np.ndarray) else merged_labels,
            class_names=class_names,
        )

    @staticmethod
    def _merge_overlapping_boxes(
        boxes: list[list[float]],
        scores: list[float],
        labels: list[int],
        iou_thresh: float = 0.20,
    ) -> tuple[list[list[float]], list[float], list[int]]:
        """Merge overlapping box duplicates into single precise bounding boxes per lesion."""
        if not boxes:
            return [], [], []

        boxes_arr = np.array(boxes, dtype=np.float32)
        scores_arr = np.array(scores, dtype=np.float32)
        labels_arr = np.array(labels, dtype=np.int32)

        merged_boxes: list[list[float]] = []
        merged_scores: list[float] = []
        merged_labels: list[int] = []

        unique_labels = np.unique(labels_arr)
        for c in unique_labels:
            mask = (labels_arr == c)
            c_boxes = boxes_arr[mask]
            c_scores = scores_arr[mask]

            order = np.argsort(-c_scores)
            c_boxes = c_boxes[order]
            c_scores = c_scores[order]

            c_boxes_list = c_boxes.tolist()
            c_scores_list = c_scores.tolist()

            while c_boxes_list:
                b_curr = c_boxes_list.pop(0)
                s_curr = c_scores_list.pop(0)

                remaining_b = []
                remaining_s = []

                for b_other, s_other in zip(c_boxes_list, c_scores_list, strict=False):
                    xi1 = max(b_curr[0], b_other[0])
                    yi1 = max(b_curr[1], b_other[1])
                    xi2 = min(b_curr[2], b_other[2])
                    yi2 = min(b_curr[3], b_other[3])
                    inter_area = max(0.0, xi2 - xi1) * max(0.0, yi2 - yi1)

                    box1_area = (b_curr[2] - b_curr[0]) * (b_curr[3] - b_curr[1])
                    box2_area = (b_other[2] - b_other[0]) * (b_other[3] - b_other[1])
                    union_area = box1_area + box2_area - inter_area
                    iou = inter_area / union_area if union_area > 0 else 0.0

                    if iou > iou_thresh:
                        b_curr[0] = min(b_curr[0], b_other[0])
                        b_curr[1] = min(b_curr[1], b_other[1])
                        b_curr[2] = max(b_curr[2], b_other[2])
                        b_curr[3] = max(b_curr[3], b_other[3])
                        s_curr = max(s_curr, s_other)
                    else:
                        remaining_b.append(b_other)
                        remaining_s.append(s_other)

                c_boxes_list = remaining_b
                c_scores_list = remaining_s

                merged_boxes.append(b_curr)
                merged_scores.append(s_curr)
                merged_labels.append(int(c))

        return merged_boxes, merged_scores, merged_labels

    def render_overlay(
        self,
        image_rgb: np.ndarray,
        result: DetectionResult,
        show_legend: bool = True,
    ) -> np.ndarray:
        """Render neat color-coded medical bounding boxes over the fundus image.

        Args:
            image_rgb: Original RGB image numpy array (H, W, 3).
            result: DetectionResult predictions.
            show_legend: Whether to append a top banner legend.

        Returns:
            Annotated RGB image numpy array (H, W, 3).
        """
        if image_rgb.dtype != np.uint8:
            img_u8 = (np.clip(image_rgb, 0, 1) * 255).astype(np.uint8)
        else:
            img_u8 = image_rgb.copy()

        canvas_bgr = cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR)

        counts = {1: 0, 2: 0, 3: 0, 4: 0}

        for box, score, label in zip(result.boxes, result.scores, result.labels, strict=False):
            color = CLASS_COLOR_PALETTE_BGR.get(label, (0, 255, 0))
            xmin, ymin, xmax, ymax = map(int, box)

            counts[label] = counts.get(label, 0) + 1

            # Draw bounding box rectangle
            cv2.rectangle(canvas_bgr, (xmin, ymin), (xmax, ymax), color, 2)

            # Draw label banner tag above box
            tag_text = f"{LESION_ID_TO_NAME.get(label, str(label))[:2]}: {score:.0%}"
            (text_w, text_h), baseline = cv2.getTextSize(
                tag_text, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1
            )
            banner_ymin = max(0, ymin - text_h - 4)
            cv2.rectangle(
                canvas_bgr,
                (xmin, banner_ymin),
                (xmin + text_w + 4, banner_ymin + text_h + 4),
                color,
                -1,
            )
            cv2.putText(
                canvas_bgr,
                tag_text,
                (xmin + 2, banner_ymin + text_h + 1),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

        if show_legend:
            h, w = canvas_bgr.shape[:2]
            banner_h = 36
            banner = np.zeros((banner_h, w, 3), dtype=np.uint8)
            banner[:] = (20, 20, 20)

            # Legend entries
            entries = [
                (CLASS_COLOR_PALETTE_BGR[1], f"EX: {counts[1]}"),
                (CLASS_COLOR_PALETTE_BGR[2], f"HE: {counts[2]}"),
                (CLASS_COLOR_PALETTE_BGR[3], f"MA: {counts[3]}"),
                (CLASS_COLOR_PALETTE_BGR[4], f"SE: {counts[4]}"),
            ]

            x_offset = 15
            for color, text in entries:
                cv2.rectangle(banner, (x_offset, 12), (x_offset + 12, 24), color, -1)
                cv2.putText(
                    banner,
                    text,
                    (x_offset + 18, 22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (220, 220, 220),
                    1,
                    cv2.LINE_AA,
                )
                x_offset += 120

            canvas_bgr = np.vstack([banner, canvas_bgr])

        return cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2RGB)
