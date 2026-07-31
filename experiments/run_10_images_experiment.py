"""Experiment script running precision FOV and mask refinement pipeline

on 10 REAL fundus images containing Class 3 (Microaneurysms, MA) and Class 4 (Soft Exudates, SE).
"""

from __future__ import annotations

from pathlib import Path
import sys
import cv2
import numpy as np

# Ensure src directory is in sys.path
project_root = Path(__file__).resolve().parent.parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from detection.xml_parser import LESION_ID_TO_NAME, parse_voc_xml
from inference.detection_predictor import DetectionResult, LesionDetectionPredictor
from inference.masked_predictor import MaskedLesionPredictor
from preprocessing.fov import FundusFOVExtractor
from visualization.mask_overlay import LESION_COLOR_PALETTE_BGR, LesionMaskVisualizer


def run_10_images_experiment():
    print("=" * 75)
    print("RUNNING PRECISION MASK EXPERIMENT ON 10 REAL FUNDUS IMAGES (CLASS 3 & 4)")
    print("=" * 75)

    xml_dir = project_root / "data" / "raw" / "lesion_detection" / "train"

    # Gather matching JPG + XML pairs
    c3_samples = []
    c4_samples = []

    for xml_file in sorted(xml_dir.glob("*.xml")):
        try:
            annot = parse_voc_xml(xml_file)
            jpg_name = xml_file.stem + ".jpg"
            found_jpgs = list((project_root / "data").rglob(jpg_name))

            if not found_jpgs:
                continue

            jpg_path = found_jpgs[0]

            if 4 in annot.labels and len(c4_samples) < 5:
                c4_samples.append((jpg_path, xml_file, annot))
            elif 3 in annot.labels and len(c3_samples) < 5:
                c3_samples.append((jpg_path, xml_file, annot))

            if len(c3_samples) >= 5 and len(c4_samples) >= 5:
                break
        except Exception:
            continue

    selected_samples = c3_samples + c4_samples
    print(f"Selected {len(selected_samples)} real fundus images with Class 3 (MA) and Class 4 (SE) annotations:")

    out_dir = project_root / "reports" / "class_3_4_batch_10"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    import torch
    from detection.detector_model import build_lesion_detector

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading REAL trained Faster R-CNN model checkpoint on device: {device}...")
    
    ckpt_path = project_root / "experiments" / "exp_faster_rcnn_lesions_v3_1536" / "checkpoints" / "best.pt"
    if not ckpt_path.exists():
        ckpt_path = project_root / "experiments" / "exp_faster_rcnn_lesions" / "checkpoints" / "best.pt"

    det_model = build_lesion_detector(num_classes=5, pretrained=False, min_size=1536, max_size=2048)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    det_model.load_state_dict(state_dict)
    det_model.to(device)
    det_model.eval()

    real_detector = LesionDetectionPredictor(
        model=det_model,
        device=device,
        score_thresh=0.20,
        target_size=(1536, 1536)
    )

    for idx, (jpg_path, xml_path, annot) in enumerate(selected_samples, 1):
        bgr = cv2.imread(str(jpg_path))
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        predictor = MaskedLesionPredictor(
            base_predictor=real_detector,
            use_fov_crop=True,
            apply_graham=False,  # Preserve natural fundus colors for human doctor display
            refine_edges=True,
        )

        res = predictor.predict_masked(rgb)

        # Render 1: Standard Natural Color Overlay
        overlay_rgb = predictor.render_overlay(
            image_rgb=rgb, result=res, show_boxes=True, show_legend=True, high_contrast_bw=False
        )

        # Render 2: High-Contrast Black & White Background Overlay for Maximum Mask Pop
        overlay_bw = predictor.render_overlay(
            image_rgb=rgb, result=res, show_boxes=True, show_legend=True, high_contrast_bw=True
        )

        out_file = out_dir / f"sample_{idx:02d}_{jpg_path.stem}.png"
        out_file_bw = out_dir / f"sample_{idx:02d}_{jpg_path.stem}_bw.png"

        cv2.imwrite(str(out_file), cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out_file_bw), cv2.cvtColor(overlay_bw, cv2.COLOR_RGB2BGR))

        classes_present = set(annot.labels)
        class_str = ", ".join([LESION_ID_TO_NAME.get(c, str(c)) for c in sorted(classes_present)])

        print(f"[{idx:02d}/10] Image: {jpg_path.name} | Resolution: {rgb.shape[1]}x{rgb.shape[0]} | Classes: {class_str} | Masks: {len(res.masks)}")
        print(f"       -> Color Overlay: {out_file.name}")
        print(f"       -> High-Contrast B&W Overlay: {out_file_bw.name}")

        summary_rows.append({
            "index": idx,
            "filename": jpg_path.name,
            "resolution": f"{rgb.shape[1]}x{rgb.shape[0]}",
            "classes": class_str,
            "box_count": len(res.masks),
            "out_path": str(out_file),
        })

    print("=" * 75)
    print(f"SUCCESS: Processed 10 images. Output directory: {out_dir}")
    print("=" * 75)
    return summary_rows


if __name__ == "__main__":
    run_10_images_experiment()
