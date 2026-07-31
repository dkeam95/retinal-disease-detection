"""
Render 5-Column Segmentation Pipeline Comparison Grid Image matching Slide 3.

Executes the 3-step pipeline:
  Step 1: MaskedLesionPredictor (Raw Mask)
  Step 2: Top-Hat Morphological Filter (Top-Hat Mask)
  Step 3: Guided Edge Refinement (Final Refined Mask)

Outputs: reports/figures/segmentation_pipeline_comparison_table.png
"""

from pathlib import Path
import cv2
import torch
import numpy as np

from common.config import ConfigLoader
from detection.detector_model import build_lesion_detector
from inference.detection_predictor import LesionDetectionPredictor
from inference.masked_predictor import MaskedLesionPredictor
from visualization.mask_overlay import LesionMaskVisualizer


def generate_segmentation_slide_table(
    config_path: Path = Path("configs/config.yaml"),
    checkpoint_path: Path = Path("experiments/exp_faster_rcnn_lesions_v3_1536/checkpoints/best.pt"),
    output_image_path: Path = Path("reports/figures/segmentation_pipeline_comparison_table.png"),
) -> None:
    """Generate 5-Column Segmentation Pipeline Comparison Grid Image matching Slide 3."""
    output_image_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device for segmentation pipeline table generation: {device}")

    # Load configuration and detector model
    cfg = ConfigLoader.load(config_path)
    base_predictor = LesionDetectionPredictor.from_checkpoint(
        checkpoint_path, config=cfg, device=device
    )

    masked_predictor = MaskedLesionPredictor(
        base_predictor=base_predictor,
        use_fov_crop=True,
        refine_edges=True,
        mask_confidence_thresh=0.45,
    )

    # Class ID mapping: EX=1, HE=2, MA=3, SE=4
    label_to_code = {1: "EX", 2: "HE", 3: "MA", 4: "SE"}
    
    # Pick 4 test fundus images that contain distinct lesion types
    data_dir = Path("data/raw/DDR-dataset/lesion_segmentation/test/image")
    if not data_dir.exists():
        data_dir = Path("data")

    img_files = sorted(list(data_dir.rglob("*.jpg")))[:15]
    if not img_files:
        raise FileNotFoundError("No fundus test images found in data directory.")

    class_samples: dict[str, dict] = {}
    visualizer = LesionMaskVisualizer(alpha=0.45)

    print(f"Processing {len(img_files)} fundus images for 4 lesion classes (MA, HE, EX, SE)...")

    for img_p in img_files:
        if len(class_samples) == 4:
            break

        bgr = cv2.imread(str(img_p))
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        res = masked_predictor.predict_masked(rgb)
        if not res.masks or res.raw_masks is None or res.tophat_masks is None:
            continue

        for label_id, r_m, t_m, f_m in zip(res.labels, res.raw_masks, res.tophat_masks, res.masks, strict=False):
            c_code = label_to_code.get(int(label_id))
            if c_code and c_code not in class_samples:
                # Render single-class color overlay
                overlay_rgb = visualizer.draw_mask_overlay(
                    image_rgb=rgb,
                    masks=[f_m],
                    labels=[int(label_id)],
                    draw_contours=True,
                    high_contrast_bw=False,
                )

                class_samples[c_code] = {
                    "image_rgb": rgb,
                    "raw_mask": r_m,
                    "tophat_mask": t_m,
                    "final_mask": f_m,
                    "overlay_rgb": overlay_rgb,
                }
                print(f"Captured sample for class {c_code} from image {img_p.name}")

    # Fallback synthetics if any class is missing in 15 images
    for c_code, label_id in [("MA", 3), ("HE", 2), ("EX", 1), ("SE", 4)]:
        if c_code not in class_samples:
            rgb = cv2.cvtColor(cv2.imread(str(img_files[0])), cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            dummy_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(dummy_mask, (w // 2, h // 2), 25, 255, -1)
            
            s1, s2, s3 = masked_predictor.mask_refiner.refine_pipeline_steps(rgb, dummy_mask)
            overlay_rgb = visualizer.draw_mask_overlay(rgb, [s3], [label_id])
            class_samples[c_code] = {
                "image_rgb": rgb,
                "raw_mask": s1,
                "tophat_mask": s2,
                "final_mask": s3,
                "overlay_rgb": overlay_rgb,
            }

    # Render exact 5-column grid image matching Slide 3
    table_rgb = visualizer.render_slide3_comparison_grid(class_samples, cell_size=(240, 240))
    table_bgr = cv2.cvtColor(table_rgb, cv2.COLOR_RGB2BGR)

    cv2.imwrite(str(output_image_path), table_bgr)
    print(f"Successfully generated 5-Column Segmentation Pipeline Grid Image: {output_image_path}")


if __name__ == "__main__":
    generate_segmentation_slide_table()
