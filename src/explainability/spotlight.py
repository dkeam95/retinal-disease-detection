"""CLAHE-Enhanced Spotlight Visualization for Retinal Disease Explainability.

Strategy:
    Standard Grad-CAM produces a blurry region heatmap that is hard to interpret
    on fundus images where pathological structures (haemorrhages, exudates) are
    small and point-like.

    This module replaces the plain JET heatmap overlay with a clinically meaningful
    *spotlight* approach:

    1. CLAHE enhancement  -- boosts local contrast so pathologies stand out.
    2. Contour detection  -- thresholds the Grad-CAM activation mask and finds
                             precise contours of high-attention regions.
    3. 4-panel report     -- side-by-side: Original | CLAHE | GradCAM | Spotlight.

Usage::

    from explainability.spotlight import SpotlightVisualizer

    viz = SpotlightVisualizer()
    panel = viz.create_4panel(
        original_rgb=img,
        heatmap=cam,
        class_name="Moderate NPDR",
        confidence=0.87,
        image_id="007-2809-100",
    )
    cv2.imwrite("spotlight.png", cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

# ── DR grade colour palette (BGR for OpenCV drawing) ──────────────────────────
# Grade 0=No DR, 1=Mild, 2=Moderate, 3=Severe, 4=Proliferative
_GRADE_COLORS_BGR: dict[int, tuple[int, int, int]] = {
    0: (120, 200, 80),  # green  — healthy
    1: (50, 200, 230),  # cyan   — mild
    2: (30, 140, 255),  # orange — moderate
    3: (0, 80, 255),  # red-orange — severe
    4: (0, 0, 220),  # red   — proliferative
}
_DEFAULT_COLOR_BGR = (0, 180, 255)  # orange fallback

_CLASS_NAMES = [
    "No DR",
    "Mild NPDR",
    "Moderate NPDR",
    "Severe NPDR",
    "Proliferative DR",
]


# ─────────────────────────────────────────────────────────────────────────────
# Core helper functions
# ─────────────────────────────────────────────────────────────────────────────


def clahe_enhance(
    image_rgb: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Apply CLAHE to the L-channel of a fundus image (LAB colour space).

    CLAHE (Contrast Limited Adaptive Histogram Equalization) boosts local contrast
    without amplifying noise globally.  Applying it to the L-channel preserves hue
    and saturation while making haemorrhages, exudates and micro-aneurysms more
    visually prominent.

    Args:
        image_rgb: Input RGB image, uint8 (H, W, 3).
        clip_limit: Contrast limiting threshold for CLAHE.
        tile_grid_size: Size of the grid for histogram equalization.

    Returns:
        CLAHE-enhanced RGB image, uint8 (H, W, 3).
    """
    if image_rgb.dtype != np.uint8:
        image_rgb = (np.clip(image_rgb, 0, 1) * 255).astype(np.uint8)

    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return enhanced


def heatmap_to_contours(
    heatmap: np.ndarray,
    threshold: float = 0.45,
    min_area: int = 50,
) -> list[np.ndarray]:
    """Convert a normalised Grad-CAM heatmap into a list of contours.

    Pipeline:
        heatmap [0,1]  ->  binary mask  ->  morphological closing  ->  findContours

    Args:
        heatmap: Normalised 2-D activation map (H, W), values in [0, 1].
        threshold: Activation threshold above which pixels are considered active.
        min_area: Minimum contour area in pixels (filters out noise contours).

    Returns:
        List of numpy contour arrays compatible with ``cv2.drawContours``.
    """
    binary = (heatmap >= threshold).astype(np.uint8) * 255

    # Morphological closing to connect nearby activated regions
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c for c in contours if cv2.contourArea(c) >= min_area]


def draw_spotlight(
    image_rgb: np.ndarray,
    heatmap: np.ndarray,
    grade_id: int = 2,
    class_name: str = "Moderate NPDR",
    confidence: float = 0.0,
    threshold: float = 0.45,
    contour_thickness: int = 2,
    show_bboxes: bool = False,
    apply_clahe: bool = True,
) -> np.ndarray:
    """Draw precise contour spotlights on a CLAHE-enhanced fundus image.

    This replaces the blurry JET heatmap overlay with:
      - CLAHE-enhanced background (sharper pathological structures)
      - Colour-coded contour outlines (one colour per DR grade)
      - Confidence + class annotation in the top-left corner
      - Optional bounding-box rectangles for each detected region

    Args:
        image_rgb: Original fundus RGB image, uint8 (H, W, 3).
        heatmap: Normalised Grad-CAM activation map (H, W), values in [0, 1].
        grade_id: Predicted DR grade index (0-4), used for contour colour.
        class_name: Human-readable class name for the annotation.
        confidence: Model confidence score for the annotation.
        threshold: Heatmap activation threshold for contour detection.
        contour_thickness: Thickness of the drawn contours.
        show_bboxes: Whether to draw bounding rectangles around contours.
        apply_clahe: Whether to apply CLAHE to the background before drawing.

    Returns:
        Annotated RGB image, uint8 (H, W, 3).
    """
    if image_rgb.dtype != np.uint8:
        image_rgb = (np.clip(image_rgb, 0, 1) * 255).astype(np.uint8)

    # Step 1: CLAHE background
    base = clahe_enhance(image_rgb) if apply_clahe else image_rgb.copy()

    # Step 2: Resize heatmap to match image dimensions
    h, w = base.shape[:2]
    hm_resized = cv2.resize(heatmap, (w, h))

    # Step 3: Find contours
    contours = heatmap_to_contours(hm_resized, threshold=threshold)

    # Step 4: Convert to BGR for OpenCV drawing
    canvas_bgr = cv2.cvtColor(base, cv2.COLOR_RGB2BGR)
    color_bgr = _GRADE_COLORS_BGR.get(grade_id, _DEFAULT_COLOR_BGR)

    if contours:
        # Draw filled semi-transparent overlay first
        overlay = canvas_bgr.copy()
        cv2.drawContours(overlay, contours, -1, color_bgr, thickness=cv2.FILLED)
        cv2.addWeighted(overlay, 0.20, canvas_bgr, 0.80, 0, canvas_bgr)

        # Draw contour outlines on top
        cv2.drawContours(
            canvas_bgr, contours, -1, color_bgr, thickness=contour_thickness
        )

        # Bounding boxes
        if show_bboxes:
            for cnt in contours:
                x, y, bw, bh = cv2.boundingRect(cnt)
                cv2.rectangle(canvas_bgr, (x, y), (x + bw, y + bh), color_bgr, 1)

    # Step 5: Annotation banner
    banner_h = 36
    banner = np.zeros((banner_h, w, 3), dtype=np.uint8)
    conf_pct = f"{confidence * 100:.1f}%" if confidence > 0 else ""
    label = f"Grade {grade_id}: {class_name}  {conf_pct}"
    cv2.putText(
        banner,
        label,
        (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color_bgr,
        2,
        cv2.LINE_AA,
    )

    result_bgr = np.vstack([banner, canvas_bgr])
    return cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)


# ─────────────────────────────────────────────────────────────────────────────
# 4-Panel Report
# ─────────────────────────────────────────────────────────────────────────────


def create_4panel(
    original_rgb: np.ndarray,
    heatmap: np.ndarray,
    grade_id: int = 2,
    class_name: str = "Moderate NPDR",
    confidence: float = 0.0,
    image_id: str = "",
    threshold: float = 0.45,
    panel_size: tuple[int, int] = (320, 320),
) -> np.ndarray:
    """Compose a 2x2 four-panel explainability report.

    Layout::

        +---------------------+---------------------+
        |   Original Image    |   CLAHE Enhanced    |
        +---------------------+---------------------+
        |   Grad-CAM Heatmap  |  Contour Spotlight  |
        +---------------------+---------------------+

    Args:
        original_rgb: Original fundus RGB image (H, W, 3).
        heatmap: Normalised Grad-CAM activation map (H, W).
        grade_id: Predicted DR grade index (0-4).
        class_name: Human-readable class name.
        confidence: Model confidence score.
        image_id: Image filename or identifier for the title bar.
        threshold: Heatmap activation threshold for spotlight contours.
        panel_size: (width, height) for each of the 4 panels.

    Returns:
        Single RGB image combining all four panels with title bars.
    """
    if original_rgb.dtype != np.uint8:
        original_rgb = (np.clip(original_rgb, 0, 1) * 255).astype(np.uint8)

    pw, ph = panel_size  # panel width, panel height
    color_bgr = _GRADE_COLORS_BGR.get(grade_id, _DEFAULT_COLOR_BGR)

    def _resize(img: np.ndarray) -> np.ndarray:
        return cv2.resize(img, (pw, ph))

    def _add_label(img_rgb: np.ndarray, title: str) -> np.ndarray:
        """Add a dark title bar below a panel."""
        bar = np.zeros((26, pw, 3), dtype=np.uint8)
        cv2.putText(
            bar,
            title,
            (6, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )
        return np.vstack([_resize(img_rgb), bar])

    # ── Panel A: Original ─────────────────────────────────────────────────
    panel_a = _add_label(original_rgb, "Original")

    # ── Panel B: CLAHE Enhanced ───────────────────────────────────────────
    clahe_img = clahe_enhance(original_rgb)
    panel_b = _add_label(clahe_img, "CLAHE Enhanced")

    # ── Panel C: Grad-CAM heatmap overlay ────────────────────────────────
    h, w = original_rgb.shape[:2]
    hm = cv2.resize(heatmap, (w, h))
    hm_uint8 = (np.clip(hm, 0, 1) * 255).astype(np.uint8)
    colored = cv2.applyColorMap(hm_uint8, cv2.COLORMAP_JET)
    colored_rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    blend = cv2.addWeighted(original_rgb, 0.55, colored_rgb, 0.45, 0)
    panel_c = _add_label(blend, "Grad-CAM Heatmap")

    # ── Panel D: Contour Spotlight ────────────────────────────────────────
    spotlight = draw_spotlight(
        image_rgb=original_rgb,
        heatmap=heatmap,
        grade_id=grade_id,
        class_name=class_name,
        confidence=confidence,
        threshold=threshold,
        show_bboxes=False,
    )
    # strip banner added by draw_spotlight (we add our own label)
    banner_h = 36
    spotlight_crop = spotlight[banner_h:, :]
    panel_d = _add_label(spotlight_crop, "Contour Spotlight")

    # ── Compose 2x2 grid ─────────────────────────────────────────────────
    row_top = np.hstack([panel_a, panel_b])
    row_bot = np.hstack([panel_c, panel_d])

    # Title bar for the whole panel
    total_w = pw * 2
    conf_str = f"{confidence * 100:.1f}%" if confidence > 0 else ""
    title_text = f"Grade {grade_id} - {class_name}  {conf_str}  |  {image_id}"
    title_bar = np.zeros((32, total_w, 3), dtype=np.uint8)
    title_bar[:] = (30, 30, 30)
    # Draw coloured grade indicator bar on the left
    title_bar[:, :6] = [color_bgr[2], color_bgr[1], color_bgr[0]]  # BGR->RGB strip
    cv2.putText(
        title_bar,
        title_text,
        (12, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (240, 240, 240),
        1,
        cv2.LINE_AA,
    )

    title_rgb = cv2.cvtColor(title_bar, cv2.COLOR_BGR2RGB)
    full_panel = np.vstack([title_rgb, row_top, row_bot])
    return full_panel


# ─────────────────────────────────────────────────────────────────────────────
# High-level visualizer class
# ─────────────────────────────────────────────────────────────────────────────


class SpotlightVisualizer:
    """High-level interface for CLAHE + Spotlight explainability visualizations.

    Supports three output modes:
        ``"spotlight"``  -- single annotated spotlight image (CLAHE + contours)
        ``"heatmap"``    -- classic Grad-CAM JET heatmap overlay
        ``"4panel"``     -- 2x2 grid: Original | CLAHE | GradCAM | Spotlight
    """

    MODES = ("spotlight", "heatmap", "4panel")

    def __init__(
        self,
        threshold: float = 0.45,
        panel_size: tuple[int, int] = (320, 320),
    ) -> None:
        """Initialize SpotlightVisualizer.

        Args:
            threshold: Activation threshold for contour detection (0–1).
            panel_size: (width, height) for each panel in 4-panel mode.
        """
        self.threshold = threshold
        self.panel_size = panel_size

    def render(
        self,
        original_rgb: np.ndarray,
        heatmap: np.ndarray,
        grade_id: int = 2,
        class_name: str = "Moderate NPDR",
        confidence: float = 0.0,
        image_id: str = "",
        mode: str = "4panel",
    ) -> np.ndarray:
        """Render visualization in the requested mode.

        Args:
            original_rgb: Original fundus RGB image (H, W, 3).
            heatmap: Normalised Grad-CAM activation (H, W), values in [0, 1].
            grade_id: Predicted DR grade index.
            class_name: Human-readable class name.
            confidence: Model confidence score (0–1).
            image_id: Image filename/stem for annotations.
            mode: One of ``"spotlight"``, ``"heatmap"``, ``"4panel"``.

        Returns:
            RGB numpy array of the rendered visualization.

        Raises:
            ValueError: If an unknown mode is given.
        """
        if mode not in self.MODES:
            raise ValueError(f"Unknown mode '{mode}'. Choose from: {self.MODES}")

        if mode == "spotlight":
            return draw_spotlight(
                image_rgb=original_rgb,
                heatmap=heatmap,
                grade_id=grade_id,
                class_name=class_name,
                confidence=confidence,
                threshold=self.threshold,
            )

        if mode == "heatmap":
            from explainability.gradcam import overlay_heatmap

            return overlay_heatmap(original_rgb, heatmap, alpha=0.5)

        # mode == "4panel"
        return create_4panel(
            original_rgb=original_rgb,
            heatmap=heatmap,
            grade_id=grade_id,
            class_name=class_name,
            confidence=confidence,
            image_id=image_id,
            threshold=self.threshold,
            panel_size=self.panel_size,
        )

    def save(
        self,
        original_rgb: np.ndarray,
        heatmap: np.ndarray,
        save_path: str | Path,
        grade_id: int = 2,
        class_name: str = "Moderate NPDR",
        confidence: float = 0.0,
        image_id: str = "",
        mode: str = "4panel",
    ) -> Path:
        """Render and save visualization to disk.

        Args:
            original_rgb: Original fundus RGB image.
            heatmap: Normalised Grad-CAM activation map.
            save_path: Output PNG file path.
            grade_id: Predicted DR grade index.
            class_name: Human-readable class name.
            confidence: Model confidence score.
            image_id: Image filename/stem for annotations.
            mode: Visualization mode (``"spotlight"``, ``"heatmap"``, ``"4panel"``).

        Returns:
            Path object of the saved PNG file.
        """
        vis = self.render(
            original_rgb=original_rgb,
            heatmap=heatmap,
            grade_id=grade_id,
            class_name=class_name,
            confidence=confidence,
            image_id=image_id,
            mode=mode,
        )
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out), bgr)
        return out
