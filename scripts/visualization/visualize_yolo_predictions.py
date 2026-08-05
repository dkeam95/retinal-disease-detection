"""Visual Result Renderer for YOLO Lesion Detection on Fundus Images.

Loads trained YOLOv8 model weights and performs full-resolution tiled inference
on fundus images, rendering color-coded bounding box overlays for 4 lesion classes:
  - EX (Hard Exudates): Yellow
  - HE (Hemorrhages): Red
  - MA (Microaneurysms): Magenta
  - SE (Soft Exudates): Cyan

Usage:
  python scripts/visualization/visualize_yolo_predictions.py --num-samples 5
  python scripts/visualization/visualize_yolo_predictions.py --image data/raw/valid/007-1774-100.jpg
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

CLASS_COLORS_BGR = {
    0: (0, 240, 255),   # EX: Yellow
    1: (0, 0, 255),     # HE: Red
    2: (255, 0, 255),   # MA: Magenta
    3: (255, 255, 0),   # SE: Cyan
}

CLASS_NAMES = {
    0: "EX (Hard Exudate)",
    1: "HE (Hemorrhage)",
    2: "MA (Microaneurysm)",
    3: "SE (Soft Exudate)",
}


def draw_bounding_boxes(
    img_bgr: np.ndarray,
    boxes: list[list[float]],
    scores: list[float],
    class_ids: list[int],
    score_thresh: float = 0.25,
) -> tuple[np.ndarray, dict[int, int]]:
    """Render medical bounding box overlays and text badges on BGR image."""
    canvas = img_bgr.copy()
    counts = {0: 0, 1: 0, 2: 0, 3: 0}

    for box, score, cls_id in zip(boxes, scores, class_ids):
        if score < score_thresh:
            continue
        counts[cls_id] = counts.get(cls_id, 0) + 1

        x1, y1, x2, y2 = map(int, box)
        color = CLASS_COLORS_BGR.get(cls_id, (0, 255, 0))

        # Thick bounding box rectangle
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

        # Label tag above box
        label_str = f"{CLASS_NAMES.get(cls_id, str(cls_id))[:2]}:{score:.0%}"
        (text_w, text_h), _ = cv2.getTextSize(label_str, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        tag_ymin = max(0, y1 - text_h - 4)

        cv2.rectangle(canvas, (x1, tag_ymin), (x1 + text_w + 4, tag_ymin + text_h + 4), color, -1)
        cv2.putText(
            canvas,
            label_str,
            (x1 + 2, tag_ymin + text_h + 1),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    # Render summary legend bar at top
    h, w = canvas.shape[:2]
    legend_h = 40
    legend_bar = np.zeros((legend_h, w, 3), dtype=np.uint8)
    legend_bar[:] = (25, 25, 25)

    offset_x = 15
    for cid, name in CLASS_NAMES.items():
        color = CLASS_COLORS_BGR[cid]
        cnt = counts.get(cid, 0)
        cv2.rectangle(legend_bar, (offset_x, 12), (offset_x + 16, 28), color, -1)
        cv2.putText(
            legend_bar,
            f"{name[:2]}: {cnt}",
            (offset_x + 22, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )
        offset_x += 140

    combined = np.vstack([legend_bar, canvas])
    return combined, counts


def predict_full_image_tiled(
    model,
    img_bgr: np.ndarray,
    tile_size: int = 1024,
    overlap: int = 256,
    score_thresh: float = 0.25,
) -> tuple[list[list[float]], list[float], list[int]]:
    """Perform full-resolution inference using overlapping tiles and Non-Maximum Suppression."""
    h, w = img_bgr.shape[:2]
    stride = tile_size - overlap

    all_boxes = []
    all_scores = []
    all_cls = []

    y_start = 0
    while y_start < h:
        y_end = min(y_start + tile_size, h)
        x_start = 0
        while x_start < w:
            x_end = min(x_start + tile_size, w)

            tile = img_bgr[y_start:y_end, x_start:x_end]
            # Inference on single tile
            results = model.predict(tile, imgsz=tile_size, conf=score_thresh, verbose=False)[0]

            if results.boxes is not None and len(results.boxes) > 0:
                for b in results.boxes:
                    xyxy = b.xyxy[0].cpu().numpy()
                    sc = float(b.conf[0].cpu().numpy())
                    cid = int(b.cls[0].cpu().numpy())

                    # Re-project tile coordinates to global full-image coordinates
                    gx1 = xyxy[0] + x_start
                    gy1 = xyxy[1] + y_start
                    gx2 = xyxy[2] + x_start
                    gy2 = xyxy[3] + y_start

                    all_boxes.append([gx1, gy1, gx2, gy2])
                    all_scores.append(sc)
                    all_cls.append(cid)

            if x_end == w:
                break
            x_start += stride
        if y_end == h:
            break
        y_start += stride

    return all_boxes, all_scores, all_cls


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize YOLO Lesion Detection Predictions")
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("experiments/exp_yolov8_lesions_1024/train_run/weights/best.pt"),
        help="Path to trained YOLO model weights (.pt)",
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Path to specific fundus image file",
    )
    parser.add_argument(
        "--val-dir",
        type=Path,
        default=Path("data/raw/valid"),
        help="Directory with validation fundus images",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of validation images to visualize",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/figures/yolo_predictions"),
        help="Output directory to save annotated result images",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Prediction confidence score threshold",
    )
    args = parser.parse_args()

    weights_path = args.weights
    if not weights_path.exists():
        # Fallback to runs directory if local weights not found
        alt_weights = Path("runs/detect/experiments/exp_yolov8_lesions_1024/train_run/weights/best.pt")
        if alt_weights.exists():
            weights_path = alt_weights
        else:
            logger.error(f"Weights file not found at: {weights_path} or {alt_weights}")
            raise SystemExit(1)

    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("ultralytics package is required. Install via `pip install ultralytics`.")
        raise SystemExit(1)

    logger.info(f"Loading trained YOLO model weights from: {weights_path}")
    model = YOLO(str(weights_path))

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine input images
    if args.image is not None and args.image.exists():
        image_files = [args.image]
    else:
        val_dir = args.val_dir
        if not val_dir.exists():
            logger.error(f"Validation image directory not found: {val_dir}")
            raise SystemExit(1)
        image_files = sorted(list(val_dir.glob("*.jpg")) + list(val_dir.glob("*.png")))[: args.num_samples]

    logger.info(f"Processing {len(image_files)} fundus images for visual overlay rendering...")

    for img_path in image_files:
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            continue

        boxes, scores, class_ids = predict_full_image_tiled(
            model=model,
            img_bgr=bgr,
            tile_size=1024,
            overlap=256,
            score_thresh=args.conf,
        )

        annotated_bgr, counts = draw_bounding_boxes(
            img_bgr=bgr,
            boxes=boxes,
            scores=scores,
            class_ids=class_ids,
            score_thresh=args.conf,
        )

        out_filename = output_dir / f"pred_{img_path.stem}.jpg"
        cv2.imwrite(str(out_filename), annotated_bgr)
        logger.info(
            f"Saved prediction overlay -> {out_filename} "
            f"(EX: {counts[0]}, HE: {counts[1]}, MA: {counts[2]}, SE: {counts[3]})"
        )

    logger.info(f"\nVisual results saved successfully to folder: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
