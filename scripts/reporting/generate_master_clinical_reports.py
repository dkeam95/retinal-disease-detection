"""Master Clinical Diagnostic & XAI/Detection Report Generator.

Generates high-resolution, uncompressed individual diagnostic cards and a master HTML report:
  1. Classification & Prediction (Grade 0-4 + Confidence)
  2. 5 SOTA Explainability Attributions (Grad-CAM, Grad-CAM++, Layer-CAM, Score-CAM, Integrated Gradients)
  3. Algorithmic Medical Lesion Marker (LesionSegmenter)
  4. Faster R-CNN Lesion Bounding Box Detection (Round 3: 1536x1536, mAP@50 = 18.24%)
"""

from __future__ import annotations

import os

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

from pathlib import Path
import sys

# Ensure src/ is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import torch

from common.config.loader import ConfigLoader
from explainability.engine import ExplainabilityEngine
from explainability.lesion_segmenter import LesionSegmenter
from inference.detection_predictor import LesionDetectionPredictor
from inference.predictor import RetinalPredictor

# ── Paths ──────────────────────────────────────────────────────────────────────
CLASSIFICATION_CONFIG = Path("configs/experiments/exp_05_convnext_tiny_v2.yaml")
CLASSIFICATION_CKPT = Path("experiments/exp_05_convnext_tiny_v2/checkpoints/best.pt")

DETECTION_CONFIG = Path("configs/detection/exp_faster_rcnn_lesions_v3_1536.yaml")
DETECTION_CKPT = Path("experiments/exp_faster_rcnn_lesions_v3_1536/checkpoints/best.pt")

TEST_XML_DIR = Path("data/raw/lesion_detection/test")
OUTPUT_DIR = Path("reports/master_clinical_reports")


def main(target_grades: list[int] | None = None, custom_images: list[str] | None = None) -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Master Clinical Diagnostic Report Generator")
    parser.add_argument("--target-grades", type=str, default=None, help="Comma-separated target DR grades to test, e.g. '3,4' or '0,1,2,3,4'")
    parser.add_argument("--images", type=str, default=None, help="Comma-separated image stems or filenames to test, e.g. '007-1774-100,007-3396-200'")
    args, _ = parser.parse_known_args()

    if args.target_grades and target_grades is None:
        target_grades = [int(g.strip()) for g in args.target_grades.split(",") if g.strip().isdigit()]
    if args.images and custom_images is None:
        custom_images = [img.strip() for img in args.images.split(",") if img.strip()]

    print("=" * 80)
    print("MASTER CLINICAL DIAGNOSTIC & XAI / DETECTION REPORT GENERATOR")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"-> Compute Device: {device}\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figures_dir = OUTPUT_DIR / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Classification Model / Ensemble & Explainability Engine
    print("[1/4] Loading Multi-Model Ensemble Classification Model & Explainability Engine...")
    cfg_cls = ConfigLoader.load(CLASSIFICATION_CONFIG)
    cls_predictor = RetinalPredictor.from_checkpoint(
        checkpoint_path=CLASSIFICATION_CKPT, config=cfg_cls, device=device
    )
    
    # Attempt loading champion ensemble if checkpoints exist
    ensemble_pairs = [
        (CLASSIFICATION_CKPT, CLASSIFICATION_CONFIG),
        (Path("experiments/exp_06_swin_tiny_v2/checkpoints/best.pt"), Path("configs/experiments/exp_06_swin_tiny_v2.yaml")),
        (Path("experiments/exp_02_densenet121_v2/checkpoints/best.pt"), Path("configs/experiments/exp_02_densenet121_v2.yaml")),
    ]
    valid_pairs = [(ckpt, cfg) for ckpt, cfg in ensemble_pairs if Path(ckpt).exists() and Path(cfg).exists()]
    if len(valid_pairs) > 1:
        from inference.ensemble import EnsemblePredictor
        print(f"  [OK] Loaded Champion {len(valid_pairs)}-Model Ensemble Predictor for maximum QWK accuracy!")
        ensemble_predictor = EnsemblePredictor.from_checkpoints(valid_pairs, device=device)
    else:
        ensemble_predictor = cls_predictor

    xai_engine = ExplainabilityEngine(model=cls_predictor.model)
    segmenter = LesionSegmenter(
        min_lesion_area=25,
        exudate_threshold_ratio=0.92,
        hemorrhage_threshold_ratio=0.45,
    )

    # 2. Load Detection Predictor (Round 3)
    print("[2/4] Loading Faster R-CNN Lesion Detector (Round 3: 1536x1536 MAX)...")
    cfg_det = ConfigLoader.load(DETECTION_CONFIG)
    det_predictor = LesionDetectionPredictor.from_checkpoint(
        checkpoint_path=DETECTION_CKPT, config=cfg_det, device=device
    )

    # Verified representative ground-truth samples for ALL 5 DR Severity Grades (Grade 0, 1, 2, 3, 4)
    grade_sample_map = {
        0: ["007-0004-000", "007-0007-000"],
        1: ["007-0033-000", "007-0096-000"],
        2: ["007-0045-000", "007-0093-000"],
        3: ["007-1723-000", "007-2289-100", "007-2766-100", "007-3286-200"],
        4: ["007-3122-100", "007-3322-200", "007-3488-200", "007-3563-200"],
    }

    test_stems = []
    if custom_images:
        for ci in custom_images:
            stem = Path(ci).stem
            test_stems.append(stem)
    elif target_grades is not None and len(target_grades) > 0:
        for tg in target_grades:
            if tg in grade_sample_map:
                test_stems.extend(grade_sample_map[tg])
    else:
        # Default: Include representative samples specifically for DR Severity Grades 2, 3, and 4
        default_target_grades = [2, 3, 4]
        for tg in default_target_grades:
            test_stems.extend(grade_sample_map[tg])

    master_reports = []

    print("\n[3/4] Generating Individual High-Resolution Diagnostic Cards...\n")
    for stem in test_stems:
        # Dynamic search across data/ for requested image stem (.jpg, .png, .jpeg)
        img_path = None
        for ext in [".jpg", ".png", ".jpeg"]:
            matches = list(Path("data").rglob(f"{stem}{ext}"))
            if matches:
                # Prefer lesion_segmentation or raw images over labels/masks
                image_matches = [m for m in matches if "label" not in str(m) and "mask" not in str(m)]
                img_path = image_matches[0] if image_matches else matches[0]
                break

        if img_path is None:
            print(f"  [SKIP] Image for {stem} not found.")
            continue

        print(f"--------------------------------------------------------------------------------")
        print(f"Processing Patient Record: {img_path.name}")

        bgr = cv2.imread(str(img_path))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        tensor = cls_predictor._prepare_image_tensor(img_path)

        # A. Classification Prediction (Ensemble)
        cls_res = ensemble_predictor.predict(img_path)
        print(f"  [Classification] Grade {cls_res.grade_id} ({cls_res.grade_name}) [Conf: {cls_res.confidence:.1%}]")

        # B. 5 SOTA XAI Attributions
        xai_outputs: dict[str, str] = {}
        xai_algos = [
            ("gradcam", "Grad-CAM", "Baseline regional feature activation"),
            ("gradcam++", "Grad-CAM++", "Higher-order gradients for multi-lesion weighting"),
            ("layer_cam", "Layer-CAM", "Fine-grained spatial gradient preservation"),
            ("score_cam", "Score-CAM", "Gradient-free perturbation confidence scoring"),
            ("integrated_gradients", "Integrated Gradients", "Pixel-level axiomatic attribution"),
        ]

        for algo_key, algo_title, algo_desc in xai_algos:
            png_name = f"{stem}_xai_{algo_key}.png"
            png_path = figures_dir / png_name
            xai_engine.save_visualization(
                input_tensor=tensor,
                original_rgb=rgb,
                save_path=png_path,
                algorithm=algo_key,
                target_class=cls_res.grade_id,
                grade_id=cls_res.grade_id,
                class_name=cls_res.grade_name,
                confidence=cls_res.confidence,
                image_id=f"{img_path.name} | {algo_title}",
                mode="4panel",
            )
            xai_outputs[algo_key] = f"figures/{png_name}"
            print(f"    - XAI Generated: {algo_title}")

        # C. Algorithmic Lesion Marker
        annotated_rgb, seg_counts = segmenter.annotate(rgb)
        seg_png_name = f"{stem}_lesion_marker.png"
        seg_png_path = figures_dir / seg_png_name
        cv2.imwrite(str(seg_png_path), cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR))
        print(f"  [Lesion Marker] HE: {seg_counts['hemorrhages']} | EX: {seg_counts['exudates']}")

        # D. Faster R-CNN Lesion Detection
        det_res = det_predictor.predict(img_path)
        det_overlay = det_predictor.render_overlay(rgb, det_res, show_legend=True)
        det_png_name = f"{stem}_faster_rcnn_detection.png"
        det_png_path = figures_dir / det_png_name
        cv2.imwrite(str(det_png_path), cv2.cvtColor(det_overlay, cv2.COLOR_RGB2BGR))
        print(f"  [Lesion Detection] {len(det_res.boxes)} bounding boxes detected")

        # Counts breakdown for detection
        det_counts = {
            "EX": det_res.class_names.count("Hard Exudate (EX)"),
            "HE": det_res.class_names.count("Hemorrhage (HE)"),
            "MA": det_res.class_names.count("Microaneurysm (MA)"),
            "SE": det_res.class_names.count("Soft Exudate (SE)"),
        }

        master_reports.append(
            {
                "image_id": img_path.name,
                "grade_id": cls_res.grade_id,
                "grade_name": cls_res.grade_name,
                "confidence": cls_res.confidence,
                "xai_paths": xai_outputs,
                "marker_path": f"figures/{seg_png_name}",
                "marker_counts": seg_counts,
                "det_path": f"figures/{det_png_name}",
                "det_boxes": len(det_res.boxes),
                "det_counts": det_counts,
            }
        )

    # 4. Generate Master HTML Report
    print("\n[4/4] Building Master HTML Executive Clinical Report...")
    html_file = OUTPUT_DIR / "master_clinical_diagnostic_report.html"

    report_sections = []
    for r in master_reports:
        report_sections.append(f"""
        <div style="background: #171923; border: 1px solid #2d3748; border-radius: 14px; padding: 25px; margin-bottom: 50px;">
            <!-- Header Banner -->
            <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 2px solid #2d3748; padding-bottom: 15px; margin-bottom: 25px;">
                <div>
                    <h2 style="color: #63b3ed; margin: 0;">📁 Patient Record: {r["image_id"]}</h2>
                    <span style="color: #a0aec0; font-size: 14px;">Classification Model: ConvNeXt-Tiny (ImageNet Pre-trained)</span>
                </div>
                <div style="text-align: right;">
                    <span style="background: #2b6cb0; color: #fff; padding: 6px 16px; border-radius: 6px; font-weight: bold; font-size: 16px;">
                        Grade {r["grade_id"]} &mdash; {r["grade_name"]} ({r["confidence"]:.1%})
                    </span>
                </div>
            </div>

            <!-- SECTION 1: AI EXPLAINABILITY SUITE (5 ALGORITHMS) -->
            <h3 style="color: #ed8936; border-left: 4px solid #ed8936; padding-left: 10px; margin-bottom: 20px;">
                👁️ 1. AI Explainability Suite (5 SOTA Attribution Algorithms)
            </h3>
            <p style="color: #cbd5e0; margin-bottom: 20px;">
                Individual 4-panel diagnostic cards showing feature activation and pixel attributions for each SOTA XAI algorithm:
            </p>

            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 30px;">
                <div style="background: #1a202c; border: 1px solid #2d3748; border-radius: 10px; padding: 12px;">
                    <h4 style="color: #90cdf4; margin-top: 0;">A. Baseline Grad-CAM</h4>
                    <img src="{r["xai_paths"]["gradcam"]}" style="width: 100%; border-radius: 6px;" alt="Grad-CAM">
                </div>
                <div style="background: #1a202c; border: 1px solid #2d3748; border-radius: 10px; padding: 12px;">
                    <h4 style="color: #90cdf4; margin-top: 0;">B. Grad-CAM++ (Higher-Order Gradients)</h4>
                    <img src="{r["xai_paths"]["gradcam++"]}" style="width: 100%; border-radius: 6px;" alt="Grad-CAM++">
                </div>
                <div style="background: #1a202c; border: 1px solid #2d3748; border-radius: 10px; padding: 12px;">
                    <h4 style="color: #90cdf4; margin-top: 0;">C. Layer-CAM (Spatial Gradient Preservation)</h4>
                    <img src="{r["xai_paths"]["layer_cam"]}" style="width: 100%; border-radius: 6px;" alt="Layer-CAM">
                </div>
                <div style="background: #1a202c; border: 1px solid #2d3748; border-radius: 10px; padding: 12px;">
                    <h4 style="color: #90cdf4; margin-top: 0;">D. Score-CAM (Gradient-Free Perturbation)</h4>
                    <img src="{r["xai_paths"]["score_cam"]}" style="width: 100%; border-radius: 6px;" alt="Score-CAM">
                </div>
            </div>

            <div style="background: #1a202c; border: 1px solid #2d3748; border-radius: 10px; padding: 12px; margin-bottom: 35px;">
                <h4 style="color: #90cdf4; margin-top: 0;">E. Integrated Gradients (Pixel-Level Axiomatic Attribution)</h4>
                <img src="{r["xai_paths"]["integrated_gradients"]}" style="width: 100%; border-radius: 6px;" alt="Integrated Gradients">
            </div>

            <!-- SECTION 2: MEDICAL LESION MARKER & SEGMENTATION -->
            <h3 style="color: #ecc94b; border-left: 4px solid #ecc94b; padding-left: 10px; margin-bottom: 20px;">
                🩺 2. Algorithmic Medical Lesion Marker (LesionSegmenter)
            </h3>
            <div style="background: #1a202c; border: 1px solid #2d3748; border-radius: 10px; padding: 15px; margin-bottom: 35px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                    <span style="color: #e2e8f0; font-weight: bold;">OpenCV Top-Hat / Bottom-Hat Heuristic Lesion Marker</span>
                    <span style="background: #744210; color: #fefcbf; padding: 4px 12px; border-radius: 4px; font-weight: bold;">
                        🔴 Hemorrhages (HE/MA): {r["marker_counts"]["hemorrhages"]} | 🟡 Exudates (EX): {r["marker_counts"]["exudates"]}
                    </span>
                </div>
                <img src="{r["marker_path"]}" style="width: 100%; border-radius: 6px;" alt="Lesion Marker">
            </div>

            <!-- SECTION 3: FASTER R-CNN LESION DETECTION -->
            <h3 style="color: #48bb78; border-left: 4px solid #48bb78; padding-left: 10px; margin-bottom: 20px;">
                🔍 3. Faster R-CNN Lesion Bounding Box Detection (Round 3: 1536x1536 MAX)
            </h3>
            <div style="background: #1a202c; border: 1px solid #2d3748; border-radius: 10px; padding: 15px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                    <div>
                        <span style="color: #e2e8f0; font-weight: bold; font-size: 16px;">Detector Model: Faster R-CNN (ResNet50-FPN)</span>
                        <span style="display: block; color: #a0aec0; font-size: 13px;">COCO Pre-trained | 1536x1536 Resolution | 4px Micro-Anchors | CLAHE LAB</span>
                    </div>
                    <div style="text-align: right;">
                        <span style="background: #276749; color: #9ae6b4; padding: 6px 14px; border-radius: 6px; font-weight: bold; font-size: 15px;">
                            Val mAP@50: 18.24% | Val mAP: 11.06%
                        </span>
                        <span style="display: block; color: #cbd5e0; font-weight: bold; margin-top: 4px;">
                            Total Detected Lesions: {r["det_boxes"]} boxes
                        </span>
                    </div>
                </div>

                <div style="display: flex; gap: 15px; margin-bottom: 15px;">
                    <span style="background: #2d3748; color: #faf089; padding: 4px 10px; border-radius: 4px; font-size: 13px;">🟡 Hard Exudate (EX): {r["det_counts"]["EX"]}</span>
                    <span style="background: #2d3748; color: #feb2b2; padding: 4px 10px; border-radius: 4px; font-size: 13px;">🔴 Hemorrhage (HE): {r["det_counts"]["HE"]}</span>
                    <span style="background: #2d3748; color: #fbb6ce; padding: 4px 10px; border-radius: 4px; font-size: 13px;">🟣 Microaneurysm (MA): {r["det_counts"]["MA"]}</span>
                    <span style="background: #2d3748; color: #e2e8f0; padding: 4px 10px; border-radius: 4px; font-size: 13px;">⚪ Soft Exudate (SE): {r["det_counts"]["SE"]}</span>
                </div>

                <img src="{r["det_path"]}" style="width: 100%; border-radius: 6px;" alt="Faster R-CNN Detection">
            </div>
        </div>
        """)

    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Master Clinical Diagnostic & XAI / Detection Report</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #0f1117; color: #e2e8f0; padding: 30px; }}
        .container {{ max-width: 1250px; margin: 0 auto; }}
        h1 {{ color: #63b3ed; border-bottom: 2px solid #4a5568; padding-bottom: 10px; }}
        .summary-card {{ background: #1a202c; border: 1px solid #2d3748; border-radius: 10px; padding: 20px; margin-bottom: 40px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🏥 Master Executive Clinical Diagnostic & AI Analytics Report</h1>
        
        <div class="summary-card">
            <h3 style="color: #90cdf4; margin-top: 0;">📋 System Overview & Model Specifications</h3>
            <ul style="color: #cbd5e0; line-height: 1.8;">
                <li><strong>Classification Backbone</strong>: ConvNeXt-Tiny (ImageNet Pre-trained) &mdash; Multi-class DR Severity Grading (Grades 0-4).</li>
                <li><strong>Explainability Suite</strong>: 5 SOTA Algorithms (Grad-CAM, Grad-CAM++, Layer-CAM, Score-CAM, Integrated Gradients) with ROI Retina Masking.</li>
                <li><strong>Lesion Marker</strong>: Algorithmic OpenCV Top-Hat/Bottom-Hat contrast detection for Hemorrhages (HE) and Exudates (EX).</li>
                <li><strong>Lesion Object Detector</strong>: Two-Stage Faster R-CNN (ResNet50-FPN) trained at $1536 \times 1536$ MAX resolution with $4\text{{px}}$ Micro-Anchors.</li>
                <li><strong>Detector Validation Metrics</strong>: <strong>Val mAP@50: 18.24%</strong> | <strong>Val mAP: 11.06%</strong> | <strong>Train Loss: 0.2389 (-51% error reduction)</strong>.</li>
            </ul>
        </div>

        {"".join(report_sections)}
    </div>
</body>
</html>
"""
    with html_file.open("w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"\n[OK] Master Clinical HTML Report saved to: {html_file}")
    print("=" * 80)
    print("MASTER CLINICAL DIAGNOSTIC REPORTS GENERATED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
