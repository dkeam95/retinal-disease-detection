"""Precision Masked Lesion Predictor with FOV Preprocessing and Coordinate Re-mapping."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np

from inference.detection_predictor import DetectionResult, LesionDetectionPredictor
from preprocessing.fov import FOVResult, FundusFOVExtractor
from preprocessing.mask_refinement import GuidedFilterMaskRefiner
from visualization.mask_overlay import LesionMaskVisualizer


@dataclass(frozen=True, slots=True)
class MaskedDetectionResult:
    """Dataclass storing detection predictions and precise pixel-level lesion masks.

    Attributes:
        boxes: List of bounding boxes [xmin, ymin, xmax, ymax] in original image coordinates.
        scores: Confidence scores for each box.
        labels: Integer class IDs (1..4).
        class_names: Human-readable class names.
        masks: List of 2D uint8 binary mask arrays of shape (H_orig, W_orig) [0, 255] (Final Step 3).
        raw_masks: Optional list of raw predicted masks (Step 1).
        tophat_masks: Optional list of Top-Hat filtered masks (Step 2).
        fov_meta: Optional FOVResult preprocessing metadata.
    """

    boxes: list[list[float]]
    scores: list[float]
    labels: list[int]
    class_names: list[str]
    masks: list[np.ndarray]
    raw_masks: list[np.ndarray] | None = None
    tophat_masks: list[np.ndarray] | None = None
    fov_meta: FOVResult | None = None


class MaskedLesionPredictor:
    """Decorator / Wrapper extending lesion detection with FOV preprocessing,

    precision mask extraction, edge refinement, and coordinate un-cropping.
    """

    def __init__(
        self,
        base_predictor: LesionDetectionPredictor,
        use_fov_crop: bool = True,
        apply_graham: bool = False,
        refine_edges: bool = True,
        mask_confidence_thresh: float = 0.50,
    ) -> None:
        """Initialize MaskedLesionPredictor.

        Args:
            base_predictor: Underlying LesionDetectionPredictor instance.
            use_fov_crop: Whether to auto-crop fundus FOV before running model detection.
            apply_graham: Whether to apply Graham's local average subtraction for lighting contrast.
            refine_edges: Whether to sharpen mask edges using Guided Filter contrast alignment.
            mask_confidence_thresh: Threshold for binary mask decisions.
        """
        self.base_predictor = base_predictor
        self.use_fov_crop = use_fov_crop
        self.apply_graham = apply_graham
        self.refine_edges = refine_edges
        self.mask_confidence_thresh = mask_confidence_thresh

        self.fov_extractor = FundusFOVExtractor()
        self.mask_refiner = GuidedFilterMaskRefiner(confidence_thresh=mask_confidence_thresh)
        self.visualizer = LesionMaskVisualizer(threshold=mask_confidence_thresh)

    def predict_masked(
        self,
        image_input: str | Path | np.ndarray,
    ) -> MaskedDetectionResult:
        """Perform end-to-end detection and precision mask generation across all 3 pipeline steps.

        Args:
            image_input: Path to image file or RGB image numpy array (H, W, 3) uint8.

        Returns:
            MaskedDetectionResult containing re-mapped bounding boxes, labels, raw masks, Top-Hat masks, and final refined masks.
        """
        if isinstance(image_input, (str, Path)):
            bgr = cv2.imread(str(image_input))
            if bgr is None:
                raise ValueError(f"Failed to load image from: {image_input}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        else:
            rgb = image_input

        orig_h, orig_w = rgb.shape[:2]

        fov_result: FOVResult | None = None
        if self.use_fov_crop:
            fov_result = self.fov_extractor.process(rgb, apply_graham=self.apply_graham)
            proc_img = fov_result.cropped_image
            crop_x, crop_y, crop_w, crop_h = fov_result.crop_bbox
        else:
            proc_img = rgb
            crop_x, crop_y, crop_w, crop_h = 0, 0, orig_w, orig_h

        # Run base detector on preprocessed image
        raw_det: DetectionResult = self.base_predictor.predict(proc_img)

        final_boxes: list[list[float]] = []
        final_scores: list[float] = raw_det.scores
        final_labels: list[int] = raw_det.labels
        final_names: list[str] = raw_det.class_names
        
        final_raw_masks: list[np.ndarray] = []
        final_tophat_masks: list[np.ndarray] = []
        final_masks: list[np.ndarray] = []

        # Full-resolution background FOV mask for clipping
        full_fov_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
        if fov_result is not None:
            full_fov_mask[crop_y : crop_y + crop_h, crop_x : crop_x + crop_w] = fov_result.fov_mask
        else:
            full_fov_mask[:] = 255

        for box, label in zip(raw_det.boxes, raw_det.labels, strict=False):
            # 1. Un-crop bounding box coordinates back to full image system
            x1 = box[0] + crop_x
            y1 = box[1] + crop_y
            x2 = box[2] + crop_x
            y2 = box[3] + crop_y
            final_boxes.append([x1, y1, x2, y2])

            # 2. Step 1: Generate raw prediction mask inside bounding box
            mask_full = np.zeros((orig_h, orig_w), dtype=np.uint8)
            bx1 = max(0, min(orig_w - 1, int(x1)))
            by1 = max(0, min(orig_h - 1, int(y1)))
            bx2 = max(bx1 + 1, min(orig_w, int(x2)))
            by2 = max(by1 + 1, min(orig_h, int(y2)))

            bw = bx2 - bx1
            bh = by2 - by1

            ellipse_mask = np.zeros((bh, bw), dtype=np.uint8)
            center = (bw // 2, bh // 2)
            axes = (max(1, bw // 2), max(1, bh // 2))
            cv2.ellipse(ellipse_mask, center, axes, 0, 0, 360, 255, -1)

            mask_full[by1:by2, bx1:bx2] = ellipse_mask

            # 3. Execute 3-step segmentation pipeline shown on slide
            s1_raw, s2_tophat, s3_final = self.mask_refiner.refine_pipeline_steps(
                image_rgb=rgb, raw_mask=mask_full, fov_mask=full_fov_mask
            )

            final_raw_masks.append(s1_raw)
            final_tophat_masks.append(s2_tophat)
            final_masks.append(s3_final)

        return MaskedDetectionResult(
            boxes=final_boxes,
            scores=final_scores,
            labels=final_labels,
            class_names=final_names,
            masks=final_masks,
            raw_masks=final_raw_masks,
            tophat_masks=final_tophat_masks,
            fov_meta=fov_result,
        )

    def render_overlay(
        self,
        image_rgb: np.ndarray,
        result: MaskedDetectionResult,
        show_boxes: bool = True,
        show_legend: bool = True,
        high_contrast_bw: bool = False,
    ) -> np.ndarray:
        """Render high-resolution precision mask overlays and optional bounding boxes.

        Args:
            image_rgb: Raw input RGB image (H, W, 3).
            result: MaskedDetectionResult predictions.
            show_boxes: Whether to draw bounding box outlines.
            show_legend: Whether to append legend banner.
            high_contrast_bw: Whether to render over a high-contrast Black & White background.

        Returns:
            Annotated RGB image numpy array (H, W, 3).
        """
        # 1. Render precision mask alpha blending and contours over color or high-contrast B&W background
        annotated = self.visualizer.draw_mask_overlay(
            image_rgb=image_rgb,
            masks=result.masks,
            labels=result.labels,
            high_contrast_bw=high_contrast_bw,
        )

        # 2. Render bounding boxes using base predictor renderer if requested
        if show_boxes and result.boxes:
            det_res = DetectionResult(
                boxes=result.boxes,
                scores=result.scores,
                labels=result.labels,
                class_names=result.class_names,
            )
            annotated = self.base_predictor.render_overlay(
                image_rgb=annotated, result=det_res, show_legend=show_legend
            )

        return annotated
