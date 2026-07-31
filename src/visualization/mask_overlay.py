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

    def render_slide3_comparison_grid(
        self,
        class_samples: dict[str, dict[str, Any]],
        cell_size: tuple[int, int] = (240, 240),
    ) -> np.ndarray:
        """Render exact 5-Column Segmentation Pipeline Comparison Grid matching Slide 3.

        Columns:
          1. Исходное изображение (Original Image)
          2. Предсказанная маска (сырая) (Raw Mask - Step 1)
          3. После Top-Hat фильтра (Top-Hat Mask - Step 2)
          4. После уточнения границ (финальная маска) (Final Refined Mask - Step 3)
          5. Наложение на изображение (Color Mask Overlay)

        Rows:
          1. Микроаневризмы (MA) - Green
          2. Кровоизлияния (HE) - Red
          3. Твёрдые экссудаты (EX) - Yellow
          4. Мягкие экссудаты (SE) - Blue

        Args:
          class_samples: Dict mapping class_code ('MA', 'HE', 'EX', 'SE') -> dict containing:
             - 'image_rgb': np.ndarray
             - 'raw_mask': np.ndarray
             - 'tophat_mask': np.ndarray
             - 'final_mask': np.ndarray
             - 'overlay_rgb': np.ndarray
          cell_size: (width, height) of individual grid cell tiles.

        Returns:
          Annotated RGB numpy array (H_total, W_total, 3).
        """
        cell_w, cell_h = cell_size
        header_h = 60
        row_label_w = 160
        num_cols = 5
        
        row_keys = ["MA", "HE", "EX", "SE"]
        row_labels_en = {
            "MA": "Microaneurysms\n(MA)",
            "HE": "Hemorrhages\n(HE)",
            "EX": "Hard Exudates\n(EX)",
            "SE": "Soft Exudates\n(SE)",
        }
        class_color_bgr = {
            "MA": (0, 220, 0),     # Green
            "HE": (0, 0, 230),     # Red
            "EX": (0, 220, 255),   # Yellow
            "SE": (255, 120, 0),   # Cyan/Blue
        }

        col_headers_en = [
            "Original\nImage",
            "Predicted Mask\n(Raw Step 1)",
            "After Top-Hat\nFilter (Step 2)",
            "After Edge Refinement\n(Final Step 3)",
            "Color Mask\nOverlay",
        ]

        total_w = row_label_w + num_cols * cell_w
        total_h = header_h + len(row_keys) * cell_h

        # Canvas with sleek dark navy background (#0f172a / BGR: 42, 23, 15)
        canvas = np.zeros((total_h, total_w, 3), dtype=np.uint8)
        canvas[:, :] = (42, 23, 15)

        # Draw Header Row
        for col_idx, text in enumerate(col_headers_en):
            x1 = row_label_w + col_idx * cell_w
            x2 = x1 + cell_w
            cv2.rectangle(canvas, (x1, 0), (x2, header_h), (60, 35, 20), -1)
            cv2.rectangle(canvas, (x1, 0), (x2, header_h), (100, 70, 40), 1)

            lines = text.split("\n")
            if len(lines) == 1:
                cv2.putText(canvas, lines[0], (x1 + 10, header_h // 2 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            else:
                cv2.putText(canvas, lines[0], (x1 + 10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (240, 240, 240), 1, cv2.LINE_AA)
                cv2.putText(canvas, lines[1], (x1 + 10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)

        # Draw Rows
        for row_idx, r_key in enumerate(row_keys):
            y1 = header_h + row_idx * cell_h
            y2 = y1 + cell_h

            # Draw Row Label Cell
            cv2.rectangle(canvas, (0, y1), (row_label_w, y2), (45, 25, 18), -1)
            cv2.rectangle(canvas, (0, y1), (row_label_w, y2), (100, 70, 40), 1)

            lbl_lines = row_labels_en[r_key].split("\n")
            lbl_color = class_color_bgr[r_key]
            cv2.putText(canvas, lbl_lines[0], (10, y1 + cell_h // 2 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.48, lbl_color, 1, cv2.LINE_AA)
            if len(lbl_lines) > 1:
                cv2.putText(canvas, lbl_lines[1], (10, y1 + cell_h // 2 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, lbl_color, 1, cv2.LINE_AA)

            sample_data = class_samples.get(r_key, {})
            img_rgb = sample_data.get("image_rgb")
            raw_mask = sample_data.get("raw_mask")
            tophat_mask = sample_data.get("tophat_mask")
            final_mask = sample_data.get("final_mask")
            overlay_rgb = sample_data.get("overlay_rgb")

            cell_images = []
            # Col 1: Original Image
            if img_rgb is not None:
                cell_images.append(cv2.cvtColor(cv2.resize(img_rgb, (cell_w, cell_h)), cv2.COLOR_RGB2BGR))
            else:
                cell_images.append(np.zeros((cell_h, cell_w, 3), dtype=np.uint8))

            # Col 2: Raw Mask
            if raw_mask is not None:
                m2 = cv2.resize(raw_mask, (cell_w, cell_h))
                cell_images.append(cv2.cvtColor(m2, cv2.COLOR_GRAY2BGR))
            else:
                cell_images.append(np.zeros((cell_h, cell_w, 3), dtype=np.uint8))

            # Col 3: Top-Hat Mask
            if tophat_mask is not None:
                m3 = cv2.resize(tophat_mask, (cell_w, cell_h))
                cell_images.append(cv2.cvtColor(m3, cv2.COLOR_GRAY2BGR))
            else:
                cell_images.append(np.zeros((cell_h, cell_w, 3), dtype=np.uint8))

            # Col 4: Final Refined Mask
            if final_mask is not None:
                m4 = cv2.resize(final_mask, (cell_w, cell_h))
                cell_images.append(cv2.cvtColor(m4, cv2.COLOR_GRAY2BGR))
            else:
                cell_images.append(np.zeros((cell_h, cell_w, 3), dtype=np.uint8))

            # Col 5: Color Overlay
            if overlay_rgb is not None:
                cell_images.append(cv2.cvtColor(cv2.resize(overlay_rgb, (cell_w, cell_h)), cv2.COLOR_RGB2BGR))
            else:
                cell_images.append(np.zeros((cell_h, cell_w, 3), dtype=np.uint8))

            # Paste 5 tiles into row
            for c_i, tile_bgr in enumerate(cell_images):
                cx1 = row_label_w + c_i * cell_w
                cx2 = cx1 + cell_w
                canvas[y1:y2, cx1:cx2] = tile_bgr
                cv2.rectangle(canvas, (cx1, y1), (cx2, y2), (80, 50, 30), 1)

        return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

    def render_xai_slide4_comparison_grid(
        self,
        xai_results: dict[str, dict[str, Any]],
        cell_size: tuple[int, int] = (240, 240),
    ) -> np.ndarray:
        """Render exact 5-Column SOTA XAI Visual Explanations Grid matching Slide 4.

        Columns (5 SOTA Algorithms):
          1. Grad-CAM
          2. Grad-CAM++
          3. Layer-CAM
          4. Score-CAM
          5. Integrated Gradients

        Rows (4 Processing Layers):
          1. Original Image (Original RGB fundus)
          2. Heatmap (JET activation heatmap)
          3. Image Overlay (Heatmap blended on fundus image)
          4. Highlighted Lesions (Lesion bounding boxes + mask overlays)

        Bottom Legend:
          - Microaneurysms (MA) [Green]
          - Hemorrhages (HE) [Red]
          - Hard Exudates (EX) [Yellow]
          - Soft Exudates / CW (SE) [Cyan/Blue]

        Args:
          xai_results: Dict mapping algo_key ('gradcam', 'gradcam++', 'layer_cam', 'score_cam', 'integrated_gradients') -> dict containing:
             - 'original_rgb': np.ndarray
             - 'heatmap_rgb': np.ndarray
             - 'overlay_rgb': np.ndarray
             - 'lesions_rgb': np.ndarray
          cell_size: (width, height) of individual tile cells.

        Returns:
          Annotated RGB numpy array (H_total, W_total, 3).
        """
        cell_w, cell_h = cell_size
        header_h = 60
        row_label_w = 160
        footer_h = 50
        num_cols = 5

        col_keys = ["gradcam", "gradcam++", "layer_cam", "score_cam", "integrated_gradients"]
        col_headers_en = [
            "Grad-CAM",
            "Grad-CAM++",
            "Layer-CAM",
            "Score-CAM",
            "Integrated\nGradients",
        ]

        row_keys = ["original", "heatmap", "overlay", "lesions"]
        row_labels_en = {
            "original": "Original\nImage",
            "heatmap": "Heatmap\n(Activation)",
            "overlay": "Image\nOverlay",
            "lesions": "Highlighted\nLesions",
        }

        total_w = row_label_w + num_cols * cell_w
        total_h = header_h + len(row_keys) * cell_h + footer_h

        # Canvas with sleek dark navy background (#0f172a / BGR: 42, 23, 15)
        canvas = np.zeros((total_h, total_w, 3), dtype=np.uint8)
        canvas[:, :] = (42, 23, 15)

        # Draw Header Row (5 Columns)
        for col_idx, (algo_k, text) in enumerate(zip(col_keys, col_headers_en, strict=False)):
            x1 = row_label_w + col_idx * cell_w
            x2 = x1 + cell_w
            cv2.rectangle(canvas, (x1, 0), (x2, header_h), (60, 35, 20), -1)
            cv2.rectangle(canvas, (x1, 0), (x2, header_h), (100, 70, 40), 1)

            lines = text.split("\n")
            if len(lines) == 1:
                cv2.putText(canvas, lines[0], (x1 + cell_w // 2 - 40, header_h // 2 + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 220, 100), 2, cv2.LINE_AA)
            else:
                cv2.putText(canvas, lines[0], (x1 + cell_w // 2 - 45, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 220, 100), 2, cv2.LINE_AA)
                cv2.putText(canvas, lines[1], (x1 + cell_w // 2 - 40, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 220, 100), 2, cv2.LINE_AA)

        # Draw 4 Rows
        for row_idx, r_key in enumerate(row_keys):
            y1 = header_h + row_idx * cell_h
            y2 = y1 + cell_h

            # Row Label Cell
            cv2.rectangle(canvas, (0, y1), (row_label_w, y2), (45, 25, 18), -1)
            cv2.rectangle(canvas, (0, y1), (row_label_w, y2), (100, 70, 40), 1)

            lbl_lines = row_labels_en[r_key].split("\n")
            cv2.putText(canvas, lbl_lines[0], (10, y1 + cell_h // 2 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (240, 240, 240), 1, cv2.LINE_AA)
            if len(lbl_lines) > 1:
                cv2.putText(canvas, lbl_lines[1], (10, y1 + cell_h // 2 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

            # Paste 5 Tiles for this Row
            for col_idx, algo_k in enumerate(col_keys):
                cx1 = row_label_w + col_idx * cell_w
                cx2 = cx1 + cell_w

                data = xai_results.get(algo_k, {})
                if r_key == "original":
                    img_rgb = data.get("original_rgb")
                elif r_key == "heatmap":
                    img_rgb = data.get("heatmap_rgb")
                elif r_key == "overlay":
                    img_rgb = data.get("overlay_rgb")
                elif r_key == "lesions":
                    img_rgb = data.get("lesions_rgb")
                else:
                    img_rgb = None

                if img_rgb is not None:
                    tile_bgr = cv2.cvtColor(cv2.resize(img_rgb, (cell_w, cell_h)), cv2.COLOR_RGB2BGR)
                else:
                    tile_bgr = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)

                canvas[y1:y2, cx1:cx2] = tile_bgr
                cv2.rectangle(canvas, (cx1, y1), (cx2, y2), (80, 50, 30), 1)

        # Draw Bottom Legend Banner
        fy1 = header_h + len(row_keys) * cell_h
        fy2 = total_h
        cv2.rectangle(canvas, (0, fy1), (total_w, fy2), (30, 18, 12), -1)
        cv2.rectangle(canvas, (0, fy1), (total_w, fy2), (100, 70, 40), 1)

        legend_items = [
            ("Microaneurysms (MA)", (0, 255, 0)),        # Green
            ("Hemorrhages (HE)", (0, 0, 255)),          # Red
            ("Hard Exudates (EX)", (0, 240, 255)),       # Yellow
            ("Soft Exudates / CW (SE)", (255, 160, 0)),  # Cyan/Blue
        ]

        leg_x = row_label_w + 20
        leg_y = fy1 + 30
        for label_text, color_bgr in legend_items:
            cv2.rectangle(canvas, (leg_x, leg_y - 12), (leg_x + 16, leg_y + 4), color_bgr, -1)
            cv2.rectangle(canvas, (leg_x, leg_y - 12), (leg_x + 16, leg_y + 4), (255, 255, 255), 1)
            cv2.putText(canvas, label_text, (leg_x + 24, leg_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1, cv2.LINE_AA)
            leg_x += 240

        return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
