"""
Optic Disc (OD) Detector & False-Positive Suppressor for Retinal Fundus Images.

Resolves the critical false-positive bottleneck where the anatomical Optic Disc
(bright circular nerve head) is misclassified as Soft Exudates (SE) or Hard Exudates (EX).
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


class OpticDiscDetector:
    """Detects Optic Disc location and suppresses false-positive exudate detections."""

    def __init__(self, expansion_factor: float = 1.25) -> None:
        """Initialize OpticDiscDetector.

        Args:
            expansion_factor: Multiplier to expand suppression radius beyond the disc edge.
        """
        self.expansion_factor = expansion_factor

    def detect(self, image_rgb: np.ndarray, fov_mask: np.ndarray | None = None) -> tuple[int, int, int] | None:
        """Detect the center coordinates (x, y) and radius r of the Optic Disc.

        Args:
            image_rgb: Input RGB image (H, W, 3).
            fov_mask: Optional binary FOV mask.

        Returns:
            Tuple (center_x, center_y, radius) or None if not found.
        """
        if image_rgb.dtype != np.uint8:
            img_uint8 = (np.clip(image_rgb, 0, 1) * 255).astype(np.uint8)
        else:
            img_uint8 = image_rgb.copy()

        # Convert to grayscale and apply FOV mask
        gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
        if fov_mask is not None:
            gray = cv2.bitwise_and(gray, gray, mask=fov_mask)

        # Gaussian blur + large morphological closing to merge vessel-divided disc components
        blurred = cv2.GaussianBlur(gray, (25, 25), 0)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
        closed = cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, kernel)

        # Adaptive thresholding for top 1.5% brightest region
        if fov_mask is not None and np.any(closed[fov_mask > 0]):
            thresh_val = np.percentile(closed[fov_mask > 0], 98.5)
        else:
            thresh_val = np.percentile(closed, 98.5)

        _, od_bin = cv2.threshold(closed, int(thresh_val), 255, cv2.THRESH_BINARY)
        if fov_mask is not None:
            od_bin = cv2.bitwise_and(od_bin, od_bin, mask=fov_mask)

        contours, _ = cv2.findContours(od_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        # Pick largest bright component
        best_cnt = max(contours, key=cv2.contourArea)
        (cx, cy), radius = cv2.minEnclosingCircle(best_cnt)
        return (int(cx), int(cy), int(radius))

    def generate_suppression_mask(self, shape: tuple[int, int], od_circle: tuple[int, int, int] | None) -> np.ndarray:
        """Generate a binary mask where Optic Disc region is 0 (suppressed) and background is 255.

        Args:
            shape: (height, width) of target mask.
            od_circle: (center_x, center_y, radius) tuple.

        Returns:
            Binary uint8 mask of shape (H, W).
        """
        h, w = shape[:2]
        mask = np.ones((h, w), dtype=np.uint8) * 255

        if od_circle is None:
            return mask

        cx, cy, r = od_circle
        suppress_radius = int(r * self.expansion_factor)
        cv2.circle(mask, (cx, cy), suppress_radius, 0, -1)
        return mask

    def filter_boxes(
        self,
        boxes: list[list[float]] | np.ndarray,
        scores: list[float] | np.ndarray,
        labels: list[int] | np.ndarray,
        od_circle: tuple[int, int, int] | None,
        bright_class_ids: set[int] = {1, 4},  # EX (1) and SE (4)
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Filter out bounding boxes of bright classes (EX, SE) located inside Optic Disc.

        Args:
            boxes: Array of shape (N, 4) [x1, y1, x2, y2].
            scores: Array of shape (N,).
            labels: Array of shape (N,).
            od_circle: Tuple (center_x, center_y, radius) of detected Optic Disc.
            bright_class_ids: Class IDs corresponding to EX (1) and SE (4).

        Returns:
            Tuple of filtered (boxes, scores, labels).
        """
        if od_circle is None or len(boxes) == 0:
            return np.array(boxes), np.array(scores), np.array(labels)

        cx, cy, r = od_circle
        suppress_radius_sq = (r * self.expansion_factor) ** 2

        keep_indices = []
        for i, (box, lbl) in enumerate(zip(boxes, labels, strict=False)):
            if int(lbl) in bright_class_ids:
                # Box center
                bx_center = (box[0] + box[2]) / 2.0
                by_center = (box[1] + box[3]) / 2.0

                dist_sq = (bx_center - cx) ** 2 + (by_center - cy) ** 2
                if dist_sq < suppress_radius_sq:
                    # Suppress box inside Optic Disc
                    continue

            keep_indices.append(i)

        if not keep_indices:
            return np.empty((0, 4)), np.empty((0,)), np.empty((0,), dtype=int)

        return np.array(boxes)[keep_indices], np.array(scores)[keep_indices], np.array(labels)[keep_indices]
