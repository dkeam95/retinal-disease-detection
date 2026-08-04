"""Evaluation Metrics for Object Detection (mAP @ IoU 0.50, mAP @ IoU 0.50:0.95)."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torchvision.ops import box_iou

from detection.xml_parser import LESION_ID_TO_NAME


def compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """Compute Average Precision using 11-point interpolation or area under PR curve."""
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))

    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1]))
    return ap


class LesionDetectionEvaluator:
    """Evaluator computing mAP (mean Average Precision) metrics for lesion bounding box detection."""

    def __init__(self, iou_thresholds: list[float] | None = None) -> None:
        """Initialize LesionDetectionEvaluator.

        Args:
            iou_thresholds: Thresholds used for mAP. Defaults to COCO's
                0.50, 0.55, ..., 0.95.
        """
        self.iou_thresholds = iou_thresholds or [round(0.50 + i * 0.05, 2) for i in range(10)]
        self.reset()

    def reset(self) -> None:
        """Reset metric state."""
        self.preds: list[dict[str, torch.Tensor]] = []
        self.targets: list[dict[str, torch.Tensor]] = []

    def update(
        self,
        preds: list[dict[str, torch.Tensor]],
        targets: list[dict[str, torch.Tensor]],
    ) -> None:
        """Update metric state with a batch of predictions and ground truth targets."""
        for p, t in zip(preds, targets, strict=False):
            self.preds.append(
                {
                    "boxes": p["boxes"].detach().cpu(),
                    "scores": p["scores"].detach().cpu(),
                    "labels": p["labels"].detach().cpu(),
                }
            )
            self.targets.append(
                {
                    "boxes": t["boxes"].detach().cpu(),
                    "labels": t["labels"].detach().cpu(),
                }
            )

    def _average_precision_for_class(self, cls_id: int, iou_threshold: float) -> float:
        """Compute AP after globally sorting predictions by confidence."""
        ground_truths: dict[int, torch.Tensor] = {}
        detections: list[tuple[float, int, torch.Tensor]] = []

        for image_id, (pred, target) in enumerate(zip(self.preds, self.targets, strict=False)):
            target_boxes = target["boxes"][target["labels"] == cls_id]
            ground_truths[image_id] = target_boxes

            pred_mask = pred["labels"] == cls_id
            for box, score in zip(
                pred["boxes"][pred_mask], pred["scores"][pred_mask], strict=False
            ):
                detections.append((float(score), image_id, box))

        n_ground_truth = sum(len(boxes) for boxes in ground_truths.values())
        if n_ground_truth == 0 or not detections:
            return 0.0

        detections.sort(key=lambda detection: detection[0], reverse=True)
        matched = {
            image_id: torch.zeros(len(boxes), dtype=torch.bool)
            for image_id, boxes in ground_truths.items()
        }
        true_positives = np.zeros(len(detections), dtype=np.float64)
        false_positives = np.zeros(len(detections), dtype=np.float64)

        for detection_id, (_, image_id, predicted_box) in enumerate(detections):
            target_boxes = ground_truths[image_id]
            if len(target_boxes) == 0:
                false_positives[detection_id] = 1.0
                continue

            overlaps = box_iou(predicted_box.unsqueeze(0), target_boxes)[0]
            best_iou, best_target = torch.max(overlaps, dim=0)
            target_id = int(best_target.item())
            if float(best_iou.item()) >= iou_threshold and not matched[image_id][target_id]:
                true_positives[detection_id] = 1.0
                matched[image_id][target_id] = True
            else:
                false_positives[detection_id] = 1.0

        cumulative_tp = np.cumsum(true_positives)
        cumulative_fp = np.cumsum(false_positives)
        recalls = cumulative_tp / float(n_ground_truth)
        precisions = cumulative_tp / np.maximum(cumulative_tp + cumulative_fp, 1e-8)
        return compute_ap(recalls, precisions)

    def compute(self) -> dict[str, float]:
        """Compute COCO-style mAP plus mAP@0.50 and mAP@0.75."""
        all_classes = set()
        for t in self.targets:
            for lbl in t["labels"].numpy():
                if lbl > 0:
                    all_classes.add(int(lbl))

        if not all_classes:
            return {"map": 0.0, "map_50": 0.0, "map_75": 0.0}

        sorted_classes = sorted(all_classes)
        ap_by_threshold = {
            threshold: {
                cls_id: self._average_precision_for_class(cls_id, threshold)
                for cls_id in sorted_classes
            }
            for threshold in set(self.iou_thresholds + [0.50, 0.75])
        }
        ap_50_per_class = ap_by_threshold[0.50]
        ap_75_per_class = ap_by_threshold[0.75]

        map_50 = float(np.mean(list(ap_50_per_class.values()))) if ap_50_per_class else 0.0
        map_75 = float(np.mean(list(ap_75_per_class.values()))) if ap_75_per_class else 0.0
        map_avg = float(
            np.mean(
                [
                    np.mean(list(ap_by_threshold[threshold].values()))
                    for threshold in self.iou_thresholds
                ]
            )
        )

        metrics_summary: dict[str, float] = {
            "map": map_avg,
            "map_50": map_50,
            "map_75": map_75,
        }

        for cls_id, ap_val in ap_50_per_class.items():
            name = LESION_ID_TO_NAME.get(cls_id, f"class_{cls_id}")
            metrics_summary[f"ap50_{name}"] = ap_val

        return metrics_summary
