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
            iou_thresholds: Optional list of IoU thresholds. Defaults to [0.50, 0.75].
        """
        self.iou_thresholds = iou_thresholds or [0.50, 0.75]
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

    def compute(self) -> dict[str, float]:
        """Compute mAP@50 and mAP@75 across all classes."""
        all_classes = set()
        for t in self.targets:
            for lbl in t["labels"].numpy():
                if lbl > 0:
                    all_classes.add(int(lbl))

        if not all_classes:
            return {"map": 0.0, "map_50": 0.0, "map_75": 0.0}

        ap_50_per_class: dict[int, float] = {}
        ap_75_per_class: dict[int, float] = {}

        for cls_id in sorted(list(all_classes)):
            for iou_thresh, ap_dict in [(0.50, ap_50_per_class), (0.75, ap_75_per_class)]:
                tp_list = []
                fp_list = []
                scores_list = []
                n_gt = 0

                for p, t in zip(self.preds, self.targets, strict=False):
                    p_mask = p["labels"] == cls_id
                    t_mask = t["labels"] == cls_id

                    p_boxes = p["boxes"][p_mask]
                    p_scores = p["scores"][p_mask]
                    t_boxes = t["boxes"][t_mask]

                    n_gt += len(t_boxes)

                    if len(p_boxes) == 0:
                        continue

                    scores_list.extend(p_scores.numpy().tolist())

                    if len(t_boxes) == 0:
                        fp_list.extend([1] * len(p_boxes))
                        tp_list.extend([0] * len(p_boxes))
                        continue

                    ious = box_iou(p_boxes, t_boxes)
                    matched_gt = set()

                    for p_idx in range(len(p_boxes)):
                        max_iou, gt_idx = torch.max(ious[p_idx], dim=0)
                        max_iou_val = float(max_iou.item())
                        gt_idx_val = int(gt_idx.item())

                        if max_iou_val >= iou_thresh and gt_idx_val not in matched_gt:
                            tp_list.append(1)
                            fp_list.append(0)
                            matched_gt.add(gt_idx_val)
                        else:
                            tp_list.append(0)
                            fp_list.append(1)

                if n_gt == 0 or len(scores_list) == 0:
                    ap_dict[cls_id] = 0.0
                    continue

                scores_np = np.array(scores_list)
                tp_np = np.array(tp_list)
                fp_np = np.array(fp_list)

                sort_indices = np.argsort(-scores_np)
                tp_sorted = tp_np[sort_indices]
                fp_sorted = fp_np[sort_indices]

                cum_tp = np.cumsum(tp_sorted)
                cum_fp = np.cumsum(fp_sorted)

                recalls = cum_tp / float(n_gt)
                precisions = cum_tp / np.maximum(cum_tp + cum_fp, 1e-8)

                ap_dict[cls_id] = compute_ap(recalls, precisions)

        map_50 = float(np.mean(list(ap_50_per_class.values()))) if ap_50_per_class else 0.0
        map_75 = float(np.mean(list(ap_75_per_class.values()))) if ap_75_per_class else 0.0
        map_avg = (map_50 + map_75) / 2.0

        metrics_summary: dict[str, float] = {
            "map": map_avg,
            "map_50": map_50,
            "map_75": map_75,
        }

        for cls_id, ap_val in ap_50_per_class.items():
            name = LESION_ID_TO_NAME.get(cls_id, f"class_{cls_id}")
            metrics_summary[f"ap50_{name}"] = ap_val

        return metrics_summary

