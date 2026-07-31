"""
Render 5-Column SOTA XAI Visual Explanations Comparison Grid Image matching Slide 4.

Algorithms Evaluated (5 Columns):
  1. Grad-CAM
  2. Grad-CAM++
  3. Layer-CAM
  4. Score-CAM
  5. Integrated Gradients

Processing Layers (4 Rows):
  1. Original Image (Original Retinal Fundus Image)
  2. Heatmap (JET Colormap Activation Map)
  3. Image Overlay (Heatmap blended on Retinal Fundus)
  4. Highlighted Lesions (Lesion bounding boxes & contours overlay)

Outputs: reports/figures/xai_pipeline_comparison_table.png
"""

import os

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

from pathlib import Path
import cv2
import numpy as np
import torch

from common.config.loader import ConfigLoader
from explainability.engine import ExplainabilityEngine
from inference.detection_predictor import LesionDetectionPredictor
from inference.masked_predictor import MaskedLesionPredictor
from inference.predictor import RetinalPredictor
from visualization.mask_overlay import LesionMaskVisualizer


def generate_xai_slide_table(
    clf_config_path: Path = Path("configs/experiments/exp_05_convnext_tiny_v2.yaml"),
    clf_ckpt_path: Path = Path("experiments/exp_05_convnext_tiny_v2/checkpoints/best.pt"),
    det_config_path: Path = Path("configs/config.yaml"),
    det_ckpt_path: Path = Path("experiments/exp_faster_rcnn_lesions_v3_1536/checkpoints/best.pt"),
    output_image_path: Path = Path("reports/figures/xai_pipeline_comparison_table.png"),
) -> None:
    """Generate 5-Column SOTA XAI Visual Explanations Comparison Grid Image matching Slide 4."""
    output_image_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device for XAI pipeline table generation: {device}")

    # Load Classification Predictor & XAI Engine
    print("[1/3] Loading ConvNeXt-Tiny & 5 SOTA XAI Algorithms...")
    cfg_clf = ConfigLoader.load(clf_config_path)
    predictor = RetinalPredictor.from_checkpoint(
        checkpoint_path=clf_ckpt_path, config=cfg_clf, device=device
    )
    engine = ExplainabilityEngine(model=predictor.model)

    # Load Detection & Mask Predictor for Row 4 (Highlighted Lesions)
    print("[2/3] Loading Faster R-CNN Lesion Detector for pathology overlay...")
    cfg_det = ConfigLoader.load(det_config_path)
    base_det_predictor = LesionDetectionPredictor.from_checkpoint(
        det_ckpt_path, config=cfg_det, device=device
    )
    masked_predictor = MaskedLesionPredictor(
        base_predictor=base_det_predictor,
        use_fov_crop=True,
        refine_edges=True,
    )

    # Pick a high-grade Class 3 / 4 sample image containing lesions
    sample_files = [
        "007-3396-200.jpg",
        "007-1829-100.jpg",
        "007-1789-100.jpg",
    ]
    img_path = None
    for sf in sample_files:
        matches = list(Path("data").rglob(sf))
        if matches:
            img_path = matches[0]
            break

    if img_path is None:
        img_path = list(Path("data").rglob("*.jpg"))[0]

    print(f"[3/3] Generating XAI heatmaps on Class 3/4 sample image: {img_path.name}")
    bgr = cv2.imread(str(img_path))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    tensor = predictor._prepare_image_tensor(img_path)

    # Run detection on image for Row 4 (Highlighted Lesions)
    det_res = masked_predictor.predict_masked(rgb)
    visualizer = LesionMaskVisualizer(alpha=0.45)
    lesions_overlay_rgb = visualizer.draw_mask_overlay(
        image_rgb=rgb,
        masks=det_res.masks if det_res.masks else [],
        labels=det_res.labels if det_res.labels else [],
        draw_contours=True,
    )

    algorithms = ["gradcam", "gradcam++", "layer_cam", "score_cam", "integrated_gradients"]
    xai_results: dict[str, dict] = {}

    for algo in algorithms:
        print(f"  - Generating heatmap for algorithm: {algo}...")
        heatmap = engine.generate_heatmap(tensor, algorithm=algo, original_rgb=rgb)

        # 1. Heatmap RGB (JET colormap)
        heatmap_u8 = (heatmap * 255).astype(np.uint8)
        heatmap_bgr = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

        # 2. Overlay RGB (Alpha blended heatmap over fundus)
        resized_rgb = cv2.resize(rgb, (heatmap.shape[1], heatmap.shape[0]))
        overlay_rgb = cv2.addWeighted(resized_rgb, 0.55, heatmap_rgb, 0.45, 0)

        # 3. Highlighted Lesions RGB (Bounding boxes & masks overlaid on heatmap overlay)
        resized_lesions = cv2.resize(lesions_overlay_rgb, (heatmap.shape[1], heatmap.shape[0]))
        lesions_rgb = cv2.addWeighted(resized_lesions, 0.65, heatmap_rgb, 0.35, 0)

        xai_results[algo] = {
            "original_rgb": resized_rgb,
            "heatmap_rgb": heatmap_rgb,
            "overlay_rgb": overlay_rgb,
            "lesions_rgb": lesions_rgb,
        }

    # Render exact 5-column grid image matching Slide 4
    table_rgb = visualizer.render_xai_slide4_comparison_grid(xai_results, cell_size=(240, 240))
    table_bgr = cv2.cvtColor(table_rgb, cv2.COLOR_RGB2BGR)

    cv2.imwrite(str(output_image_path), table_bgr)
    print(f"Successfully generated 5-Column SOTA XAI Grid Image: {output_image_path}")


if __name__ == "__main__":
    generate_xai_slide_table()
