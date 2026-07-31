"""Unit tests for precision masking, FOV extraction, edge refinement, and masked predictor re-mapping."""

from __future__ import annotations

from unittest.mock import MagicMock
import numpy as np
import pytest

from inference.detection_predictor import DetectionResult, LesionDetectionPredictor
from inference.masked_predictor import MaskedDetectionResult, MaskedLesionPredictor
from preprocessing.fov import FOVResult, FundusFOVExtractor
from preprocessing.mask_refinement import GuidedFilterMaskRefiner, refine_lesion_mask
from visualization.mask_overlay import LESION_COLOR_PALETTE_BGR, LesionMaskVisualizer


@pytest.fixture
def dummy_fundus_image() -> np.ndarray:
    """Generate a synthetic fundus RGB image (H=400, W=400, 3) with a central bright circle."""
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    # Draw retina circle (center 200, 200, radius 150)
    import cv2

    cv2.circle(img, (200, 200), 150, (180, 100, 40), -1)
    # Add a mock lesion (bright yellow dot for EX)
    cv2.circle(img, (220, 210), 15, (255, 255, 0), -1)
    return img


def test_fundus_fov_extractor(dummy_fundus_image: np.ndarray) -> None:
    """Test FOV mask extraction, ROI cropping, and Graham preprocessing."""
    extractor = FundusFOVExtractor(black_threshold=10, margin_pct=0.02, circle_fit=True)
    res: FOVResult = extractor.process(dummy_fundus_image, apply_graham=True)

    assert isinstance(res, FOVResult)
    assert res.orig_shape == (400, 400)
    assert res.cropped_image.shape[2] == 3
    assert res.fov_mask.shape == res.cropped_image.shape[:2]
    # Crop bbox should be approximately around the 300px circle (x1 ~ 50, y1 ~ 50, w ~ 300, h ~ 300)
    x, y, w, h = res.crop_bbox
    assert 0 <= x < 100
    assert 0 <= y < 100
    assert 250 <= w <= 400
    assert 250 <= h <= 400


def test_mask_refinement(dummy_fundus_image: np.ndarray) -> None:
    """Test guided filter mask edge refinement."""
    refiner = GuidedFilterMaskRefiner(confidence_thresh=0.5)

    raw_mask = np.zeros((400, 400), dtype=np.float32)
    # Soft blurry mask around (220, 210)
    import cv2

    cv2.circle(raw_mask, (220, 210), 20, 0.8, -1)
    raw_mask = cv2.GaussianBlur(raw_mask, (9, 9), 0)

    fov_mask = np.zeros((400, 400), dtype=np.uint8)
    cv2.circle(fov_mask, (200, 200), 150, 255, -1)

    refined = refiner.refine(dummy_fundus_image, raw_mask, fov_mask=fov_mask)
    assert refined.shape == (400, 400)
    assert refined.dtype == np.uint8
    assert np.max(refined) == 255

    # Check standalone helper
    refined_helper = refine_lesion_mask(dummy_fundus_image, raw_mask, fov_mask=fov_mask)
    assert refined_helper.shape == (400, 400)


def test_lesion_mask_visualizer(dummy_fundus_image: np.ndarray) -> None:
    """Test multi-class alpha blending and side-by-side visualization."""
    vis = LesionMaskVisualizer(alpha=0.4, contour_thickness=2)

    mask1 = np.zeros((400, 400), dtype=np.uint8)
    import cv2

    cv2.circle(mask1, (220, 210), 15, 255, -1)

    # 1. Overlay single mask
    overlay = vis.draw_mask_overlay(dummy_fundus_image, masks=[mask1], labels=[1])
    assert overlay.shape == dummy_fundus_image.shape
    assert overlay.dtype == np.uint8

    # 2. Side-by-side view
    sbs = vis.draw_side_by_side(dummy_fundus_image, masks=[mask1], labels=[1])
    assert sbs.shape == (400, 800, 3)


def test_masked_lesion_predictor(dummy_fundus_image: np.ndarray) -> None:
    """Test MaskedLesionPredictor end-to-end with a mocked base detector."""
    mock_base = MagicMock(spec=LesionDetectionPredictor)

    # Mock base predictor to return 1 detection box inside the cropped ROI
    mock_base.predict.return_value = DetectionResult(
        boxes=[[50.0, 50.0, 100.0, 100.0]],
        scores=[0.92],
        labels=[1],
        class_names=["Hard Exudate"],
    )
    mock_base.render_overlay.side_effect = lambda image_rgb, result, show_legend: image_rgb

    masked_pred = MaskedLesionPredictor(
        base_predictor=mock_base,
        use_fov_crop=True,
        apply_graham=False,
        refine_edges=True,
    )

    res: MaskedDetectionResult = masked_pred.predict_masked(dummy_fundus_image)

    assert isinstance(res, MaskedDetectionResult)
    assert len(res.boxes) == 1
    assert len(res.masks) == 1
    assert res.masks[0].shape == (400, 400)

    # Bounding box should be shifted by crop_bbox offset
    assert res.fov_meta is not None
    crop_x, crop_y, _, _ = res.fov_meta.crop_bbox
    assert res.boxes[0][0] == 50.0 + crop_x
    assert res.boxes[0][1] == 50.0 + crop_y

    # Test overlay rendering
    rendered = masked_pred.render_overlay(dummy_fundus_image, res, show_boxes=True)
    assert rendered.shape == dummy_fundus_image.shape
    assert rendered.dtype == np.uint8
