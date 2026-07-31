"""Precision Lesion Mask and FOV Visualizer.

Provides high-resolution alpha-blending, anti-aliased contour stroke rendering,
and multi-class color palette overlays for medical image inspection.
"""

from __future__ import annotations

import cv2
import numpy as np

# Color palette for medical lesion visualization (BGR format)
LESION_COLOR_PALETTE_BGR: dict[int, tuple[int, int, int]] = {
    1: (0, 240, 255),  # EX (Hard Exudates) -> Yellow
    2: (0, 0, 255),    # HE (Hemorrhages) -> Red
    3: (255, 0, 255),  # MA (Microaneurysms) -> Magenta
    4: (255, 255, 0),  # SE (Soft Exudates) -> Cyan
    5: (0, 255, 0),    # Optic Disc / FOV Boundary -> Green
}


class LesionMaskVisualizer:
    """Renders precise semi-transparent alpha overlays and sharp contour strokes."""

    def __init__(
        self,
        alpha: float = 0.35,
        contour_thickness: int = 2,
        threshold: float = 0.50,
    ) -> None:
        """Initialize LesionMaskVisualizer.

        Args:
            alpha: Transparency factor for filled mask regions (0.0 to 1.0).
            contour_thickness: Line thickness in pixels for border contour strokes.
            threshold: Probability threshold for binarizing continuous prediction masks.
        """
        self.alpha = alpha
        self.contour_thickness = contour_thickness
        self.threshold = threshold

    def draw_mask_overlay(
        self,
        image_rgb: np.ndarray,
        masks: list[np.ndarray] | dict[int, np.ndarray] | np.ndarray,
        labels: list[int] | np.ndarray | None = None,
        draw_contours: bool = True,
        high_contrast_bw: bool = False,
        clahe_clip_limit: float = 3.0,
    ) -> np.ndarray:
        """Render multi-class lesion masks with alpha blending and boundary contours.

        Args:
            image_rgb: Raw input RGB image (H, W, 3) uint8.
            masks: Single mask array (H, W), list of binary masks, or dict mapping class_id -> mask array.
            labels: Class IDs corresponding to masks if masks is a list.
            draw_contours: Whether to draw anti-aliased border contours around masks.
            high_contrast_bw: Whether to convert background fundus image to high-contrast B&W for maximum mask pop.
            clahe_clip_limit: CLAHE contrast clip limit for B&W enhancement.

        Returns:
            Annotated RGB image numpy array (H, W, 3) uint8.
        """
        if image_rgb.dtype != np.uint8:
            img_u8 = (np.clip(image_rgb, 0, 1) * 255).astype(np.uint8)
        else:
            img_u8 = image_rgb

        if high_contrast_bw:
            # Convert to green channel B&W with CLAHE contrast enhancement
            green_ch = img_u8[:, :, 1]
            clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=(8, 8))
            bw_enhanced = clahe.apply(green_ch)
            canvas_bgr = cv2.cvtColor(bw_enhanced, cv2.COLOR_GRAY2BGR)
        else:
            canvas_bgr = cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR)

        h, w = canvas_bgr.shape[:2]

        # Standardize mask inputs to list of (mask_array, label_id)
        mask_items: list[tuple[np.ndarray, int]] = []

        if isinstance(masks, dict):
            for label_id, m_arr in masks.items():
                mask_items.append((m_arr, label_id))
        elif isinstance(masks, list):
            labels_list = labels if labels is not None else list(range(1, len(masks) + 1))
            for m_arr, label_id in zip(masks, labels_list, strict=False):
                mask_items.append((m_arr, label_id))
        elif isinstance(masks, np.ndarray):
            if masks.ndim == 2:
                mask_items.append((masks, 1 if labels is None else int(labels[0])))
            elif masks.ndim == 3:
                # Shape (N, H, W)
                labels_list = labels if labels is not None else list(range(1, masks.shape[0] + 1))
                for i in range(masks.shape[0]):
                    mask_items.append((masks[i], int(labels_list[i])))

        # Create blended layer
        overlay = canvas_bgr.copy()

        for mask_arr, label_id in mask_items:
            if mask_arr.shape[:2] != (h, w):
                mask_arr = cv2.resize(mask_arr, (w, h), interpolation=cv2.INTER_NEAREST)

            if mask_arr.dtype == np.float32 or mask_arr.dtype == np.float64:
                binary_mask = (mask_arr >= self.threshold).astype(np.uint8)
            else:
                binary_mask = (mask_arr > 0).astype(np.uint8)

            if not np.any(binary_mask):
                continue

            color_bgr = LESION_COLOR_PALETTE_BGR.get(label_id, (0, 255, 0))

            # 1. Fill mask region with alpha blending
            roi_idx = binary_mask == 1
            colored_fill = np.zeros_like(canvas_bgr, dtype=np.uint8)
            colored_fill[roi_idx] = color_bgr

            overlay[roi_idx] = cv2.addWeighted(
                canvas_bgr[roi_idx], 1.0 - self.alpha, colored_fill[roi_idx], self.alpha, 0
            )

            # 2. Draw sharp anti-aliased contour border stroke
            if draw_contours:
                contours, _ = cv2.findContours(
                    binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(
                    overlay,
                    contours,
                    -1,
                    color_bgr,
                    thickness=self.contour_thickness,
                    lineType=cv2.LINE_AA,
                )

        return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    def draw_side_by_side(
        self,
        image_rgb: np.ndarray,
        masks: list[np.ndarray] | dict[int, np.ndarray] | np.ndarray,
        labels: list[int] | np.ndarray | None = None,
    ) -> np.ndarray:
        """Create side-by-side comparison of raw fundus image and mask overlay.

        Returns:
            Concatenated RGB image array of shape (H, W*2, 3).
        """
        overlay_rgb = self.draw_mask_overlay(image_rgb, masks, labels=labels)
        if image_rgb.dtype != np.uint8:
            orig_u8 = (np.clip(image_rgb, 0, 1) * 255).astype(np.uint8)
        else:
            orig_u8 = image_rgb
        return np.hstack([orig_u8, overlay_rgb])
