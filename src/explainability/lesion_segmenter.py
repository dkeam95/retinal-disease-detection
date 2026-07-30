"""Clinical Lesion Segmenter & Marker for Retinal Fundus Images.

Implements automated detection and clinical color-coded marking of Diabetic Retinopathy lesions:
  - 🔴 Red circles: Hemorrhages & Microaneurysms (HE / MA - dark lesions)
  - 🟡 Yellow circles: Hard Exudates (EX - bright yellowish lipid deposits)

Uses Green-channel contrast extraction, CLAHE LAB-space filtering, and Top/Bottom-Hat
morphological operators combined with adaptive blob contour detection.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


class LesionSegmenter:
    """Clinical lesion segmenter and overlay marker for fundus images."""

    def __init__(
        self,
        min_lesion_area: int = 12,
        max_lesion_area: int = 2500,
        exudate_threshold_ratio: float = 0.82,
        hemorrhage_threshold_ratio: float = 0.35,
    ) -> None:
        """Initialize LesionSegmenter.

        Args:
            min_lesion_area: Minimum pixel area to classify as a valid lesion contour.
            max_lesion_area: Maximum pixel area to exclude large anatomical structures (e.g. optic disc).
            exudate_threshold_ratio: Percentile threshold for bright exudates.
            hemorrhage_threshold_ratio: Percentile threshold for dark hemorrhages.
        """
        self.min_lesion_area = min_lesion_area
        self.max_lesion_area = max_lesion_area
        self.exudate_threshold_ratio = exudate_threshold_ratio
        self.hemorrhage_threshold_ratio = hemorrhage_threshold_ratio

    def _get_retina_fov_mask(self, image_rgb: np.ndarray) -> np.ndarray:
        """Generate a clean binary mask of the circular retinal Field of View (FOV)."""
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)

        # Fill any internal holes
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

        # Erode border by 15px to eliminate edge boundary noise at the retina rim
        kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        return cv2.erode(mask, kernel_erode)

    def detect_exudates(self, image_rgb: np.ndarray) -> list[dict[str, Any]]:
        """Detect Hard Exudates (EX - bright yellowish lipid deposits).

        Returns:
            List of dictionaries containing contour, bounding circle (center_x, center_y, radius), and area.
        """
        if image_rgb.dtype != np.uint8:
            image_rgb = (np.clip(image_rgb, 0, 1) * 255).astype(np.uint8)

        # Generate Retina FOV mask
        fov_mask = self._get_retina_fov_mask(image_rgb)

        # Convert to LAB & HSV color spaces
        lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
        l_channel = lab[:, :, 0]

        # Apply CLAHE to L-channel
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced_l = clahe.apply(l_channel)

        # Morphological Top-Hat to isolate bright structures smaller than optic disc
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        top_hat = cv2.morphologyEx(enhanced_l, cv2.MORPH_TOPHAT, kernel)
        top_hat = cv2.bitwise_and(top_hat, top_hat, mask=fov_mask)

        # Adaptive thresholding for bright spots
        thresh_val = (
            np.percentile(top_hat[top_hat > 0], self.exudate_threshold_ratio * 100)
            if np.any(top_hat > 0)
            else 180
        )
        _, binary_ex = cv2.threshold(top_hat, int(thresh_val), 255, cv2.THRESH_BINARY)
        binary_ex = cv2.bitwise_and(binary_ex, binary_ex, mask=fov_mask)

        # Exclude optic disc region (largest bright blob)
        contours, _ = cv2.findContours(
            binary_ex, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        exudates = []

        if contours:
            # Sort contours by area
            sorted_cnts = sorted(contours, key=cv2.contourArea, reverse=True)
            # Skip the largest contour if it resembles the optic disc (> 3000 px)
            for cnt in sorted_cnts:
                area = cv2.contourArea(cnt)
                if self.min_lesion_area <= area <= self.max_lesion_area:
                    (x, y), radius = cv2.minEnclosingCircle(cnt)
                    exudates.append(
                        {
                            "type": "exudate",
                            "contour": cnt,
                            "center": (int(x), int(y)),
                            "radius": max(int(radius), 4),
                            "area": float(area),
                        }
                    )

        return exudates

    def detect_hemorrhages(self, image_rgb: np.ndarray) -> list[dict[str, Any]]:
        """Detect Hemorrhages & Microaneurysms (HE / MA - dark lesions).

        Returns:
            List of dictionaries containing contour, bounding circle (center_x, center_y, radius), and area.
        """
        if image_rgb.dtype != np.uint8:
            image_rgb = (np.clip(image_rgb, 0, 1) * 255).astype(np.uint8)

        # Generate Retina FOV mask
        fov_mask = self._get_retina_fov_mask(image_rgb)

        # Extract Green channel (maximum absorption / contrast for hemoglobin)
        g_channel = image_rgb[:, :, 1]

        # Apply CLAHE to Green channel
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced_g = clahe.apply(g_channel)

        # Morphological Bottom-Hat (black top-hat) to isolate dark round structures
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        bottom_hat = cv2.morphologyEx(enhanced_g, cv2.MORPH_BLACKHAT, kernel)
        bottom_hat = cv2.bitwise_and(bottom_hat, bottom_hat, mask=fov_mask)

        # Threshold dark lesions
        thresh_val = (
            np.percentile(
                bottom_hat[bottom_hat > 0],
                (1.0 - self.hemorrhage_threshold_ratio) * 100,
            )
            if np.any(bottom_hat > 0)
            else 35
        )
        _, binary_he = cv2.threshold(
            bottom_hat, int(thresh_val), 255, cv2.THRESH_BINARY
        )
        binary_he = cv2.bitwise_and(binary_he, binary_he, mask=fov_mask)

        # Morphological closing to join split blood spots
        binary_he = cv2.morphologyEx(
            binary_he,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        )

        contours, _ = cv2.findContours(
            binary_he, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        hemorrhages = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if self.min_lesion_area <= area <= self.max_lesion_area:
                # Check circularity to distinguish round lesions from linear main blood vessels
                perimeter = cv2.arcLength(cnt, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * (area / (perimeter * perimeter))
                    if circularity >= 0.25:  # filter out long linear vessel lines
                        (x, y), radius = cv2.minEnclosingCircle(cnt)
                        hemorrhages.append(
                            {
                                "type": "hemorrhage",
                                "contour": cnt,
                                "center": (int(x), int(y)),
                                "radius": max(int(radius), 4),
                                "area": float(area),
                            }
                        )

        return hemorrhages

    def annotate(
        self,
        image_rgb: np.ndarray,
        draw_exudates: bool = True,
        draw_hemorrhages: bool = True,
        show_legend: bool = True,
    ) -> tuple[np.ndarray, dict[str, int]]:
        """Detect lesions and render neat medical annotations on top of the fundus image.

        - 🔴 Red circles: Hemorrhages (HE / MA)
        - 🟡 Yellow circles: Hard Exudates (EX)

        Args:
            image_rgb: Original fundus RGB image (H, W, 3).
            draw_exudates: Whether to annotate Hard Exudates.
            draw_hemorrhages: Whether to annotate Hemorrhages.
            show_legend: Whether to add a clinical legend banner at the top.

        Returns:
            Tuple of (Annotated RGB image, dict with count of detected lesions).
        """
        if image_rgb.dtype != np.uint8:
            img_uint8 = (np.clip(image_rgb, 0, 1) * 255).astype(np.uint8)
        else:
            img_uint8 = image_rgb.copy()

        canvas_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
        counts = {"hemorrhages": 0, "exudates": 0}

        # 1. Detect Hemorrhages (HE)
        if draw_hemorrhages:
            hemorrhages = self.detect_hemorrhages(img_uint8)
            counts["hemorrhages"] = len(hemorrhages)

            # Draw semi-transparent red overlays
            overlay = canvas_bgr.copy()
            for item in hemorrhages:
                cv2.circle(
                    overlay, item["center"], item["radius"], (0, 0, 220), -1
                )  # BGR Red fill
            cv2.addWeighted(overlay, 0.35, canvas_bgr, 0.65, 0, canvas_bgr)

            # Draw solid red outline circles
            for item in hemorrhages:
                cv2.circle(
                    canvas_bgr, item["center"], item["radius"], (0, 0, 255), 2
                )  # Red circle outline
                cv2.circle(
                    canvas_bgr, item["center"], 1, (0, 0, 255), -1
                )  # Center point

        # 2. Detect Hard Exudates (EX)
        if draw_exudates:
            exudates = self.detect_exudates(img_uint8)
            counts["exudates"] = len(exudates)

            # Draw semi-transparent yellow overlays
            overlay = canvas_bgr.copy()
            for item in exudates:
                cv2.circle(
                    overlay, item["center"], item["radius"], (0, 220, 255), -1
                )  # BGR Yellow fill
            cv2.addWeighted(overlay, 0.35, canvas_bgr, 0.65, 0, canvas_bgr)

            # Draw solid yellow outline circles
            for item in exudates:
                cv2.circle(
                    canvas_bgr, item["center"], item["radius"], (0, 240, 255), 2
                )  # Yellow circle outline
                cv2.circle(
                    canvas_bgr, item["center"], 1, (0, 240, 255), -1
                )  # Center point

        # 3. Add Legend Banner
        if show_legend:
            h, w = canvas_bgr.shape[:2]
            banner_h = 36
            banner = np.zeros((banner_h, w, 3), dtype=np.uint8)
            banner[:] = (20, 20, 20)

            # Draw Red indicator dot for Hemorrhages
            cv2.circle(banner, (18, 18), 6, (0, 0, 255), -1)
            cv2.putText(
                banner,
                f"Hemorrhages (HE/MA): {counts['hemorrhages']}",
                (30, 23),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (220, 220, 220),
                1,
                cv2.LINE_AA,
            )

            # Draw Yellow indicator dot for Exudates
            cv2.circle(banner, (240, 18), 6, (0, 240, 255), -1)
            cv2.putText(
                banner,
                f"Hard Exudates (EX): {counts['exudates']}",
                (252, 23),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (220, 220, 220),
                1,
                cv2.LINE_AA,
            )

            canvas_bgr = np.vstack([banner, canvas_bgr])

        return cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2RGB), counts
