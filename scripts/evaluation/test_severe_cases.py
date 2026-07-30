"""Targeted test script for Grade 3 (Severe NPDR) and Grade 4 (Proliferative DR) fundus images.

Runs Ensemble inference and generates CLAHE + Spotlight 4-panel explainability reports.
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
from explainability.batch_audit import GradCAMBatchAuditor
from explainability.spotlight import SpotlightVisualizer
from inference.ensemble import EnsemblePredictor
from inference.predictor import RetinalPredictor

# ── Setup Paths & Models ───────────────────────────────────────────────────────
CONFIG_DIR = Path("configs/experiments")
EXPERIMENTS_DIR = Path("experiments")
DATA_DIR = Path("data/raw/test")
OUTPUT_DIR = Path("reports/figures/severe_cases_audit")

MODELS = [
    (
        "exp_05_convnext_tiny_v2",
        EXPERIMENTS_DIR / "exp_05_convnext_tiny_v2" / "checkpoints" / "best.pt",
        0.40,
    ),
    (
        "exp_07_swin_t_v2",
        EXPERIMENTS_DIR / "exp_07_swin_t_v2" / "checkpoints" / "best.pt",
        0.35,
    ),
    (
        "exp_02_densenet121_v2",
        EXPERIMENTS_DIR / "exp_02_densenet121_v2" / "checkpoints" / "best.pt",
        0.25,
    ),
]

TEST_SAMPLES = [
    # (Filename, Ground Truth Grade, Description)
    ("007-5988-300.jpg", 3, "Severe NPDR Sample 1"),
    ("007-5999-300.jpg", 3, "Severe NPDR Sample 2"),
    ("007-6961-400.jpg", 4, "Proliferative DR Sample 1"),
    ("007-6962-400.jpg", 4, "Proliferative DR Sample 2"),
]


def main() -> None:
    print("=" * 75)
    print(
        "SEVERE RETINAL DISEASE EVALUATION (GRADE 3 - SEVERE & GRADE 4 - PROLIFERATIVE)"
    )
    print("=" * 75)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"-> Compute Device: {device}\n")

    # 1. Load Ensemble
    print("[1/3] Loading 3-Model Ensemble...")
    predictors = []
    weights = []
    for exp_name, ckpt_path, weight in MODELS:
        cfg_path = CONFIG_DIR / f"{exp_name}.yaml"
        cfg = ConfigLoader.load(cfg_path)
        pred = RetinalPredictor.from_checkpoint(
            checkpoint_path=ckpt_path, config=cfg, device=device
        )
        predictors.append(pred)
        weights.append(weight)
        print(f"  [OK] Loaded {cfg.model.architecture:<20} from {ckpt_path}")

    ensemble = EnsemblePredictor(predictors=predictors, weights=weights)
    print("-> Ensemble initialized!\n")

    # 2. Run Inference & Generate CLAHE Spotlight Visualizations
    print("[2/3] Running Inference & Generating CLAHE + Spotlight 4-Panel Reports...\n")
    visualizer = SpotlightVisualizer(threshold=0.40, panel_size=(360, 360))
    auditor = GradCAMBatchAuditor(predictor=predictors[0], output_dir=OUTPUT_DIR)

    results_summary = []

    for img_name, gt_label, desc in TEST_SAMPLES:
        img_path = DATA_DIR / img_name
        if not img_path.exists():
            print(f"  [SKIP] Image not found: {img_path}")
            continue

        # Predict
        res = ensemble.predict(img_path)
        is_correct = res.grade_id == gt_label
        status_str = "MATCH [SUCCESS]" if is_correct else "DISCREPANCY [WARNING]"

        print(
            f"-----------------------------------------------------------------------"
        )
        print(f"File         : {img_name} ({desc})")
        print(f"Ground Truth : Grade {gt_label} ({ensemble.class_names[gt_label]})")
        print(f"Prediction   : Grade {res.grade_id} ({res.grade_name})  [{status_str}]")
        print(f"Confidence   : {res.confidence:.2%}")
        print("Distribution :")
        for cls_id, (cname, prob) in enumerate(
            zip(ensemble.class_names, res.probabilities)
        ):
            bar = "#" * int(prob * 30)
            mark = " <--" if cls_id == res.grade_id else ""
            print(f"   Class {cls_id} ({cname:<18}): {prob * 100:6.2f}% | {bar}{mark}")

        # Load image RGB & Tensor for Grad-CAM
        bgr = cv2.imread(str(img_path))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        tensor = predictors[0]._prepare_image_tensor(img_path)

        # Compute Grad-CAM heatmap
        heatmap = auditor.gradcam.generate(
            input_tensor=tensor, target_class=res.grade_id
        )

        # Save 4-panel spotlight report
        out_png = (
            OUTPUT_DIR
            / f"severe_grade{gt_label}_pred{res.grade_id}_{img_path.stem}_4panel.png"
        )
        visualizer.save(
            original_rgb=rgb,
            heatmap=heatmap,
            save_path=out_png,
            grade_id=res.grade_id,
            class_name=res.grade_name,
            confidence=res.confidence,
            image_id=img_path.stem,
            mode="4panel",
        )
        print(f"Saved 4-Panel Report -> {out_png}\n")

        results_summary.append(
            {
                "image_id": img_name,
                "gt_label": gt_label,
                "pred_label": res.grade_id,
                "confidence": res.confidence,
                "is_correct": is_correct,
                "report_path": str(out_png),
            }
        )

    # 3. Export HTML Gallery for Severe Cases
    print("[3/3] Exporting HTML Gallery...")
    html_file = OUTPUT_DIR / "severe_cases_gallery.html"

    cards = []
    for r in results_summary:
        rel_path = Path(r["report_path"]).name
        color = "#28a745" if r["is_correct"] else "#dc3545"
        icon = "V" if r["is_correct"] else "X"
        cards.append(f"""
        <div class="card" style="border-left: 6px solid {color}; margin-bottom: 25px; background: #1a202c; border-radius: 8px; overflow: hidden;">
            <div style="padding: 15px; background: #2d3748; color: #fff; display: flex; justify-content: space-between; align-items: center;">
                <h3 style="margin: 0;">{r["image_id"]} &mdash; GT: Grade {r["gt_label"]} vs Pred: Grade {r["pred_label"]}</h3>
                <span style="background: {color}; color: white; padding: 4px 12px; border-radius: 4px; font-weight: bold;">{icon} Conf: {r["confidence"]:.1%}</span>
            </div>
            <img src="{rel_path}" style="width: 100%; display: block;" alt="Severe Case Spotlight">
        </div>
        """)

    gallery_html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Severe DR Cases Evaluation (Grade 3 & 4)</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #0f1117; color: #e2e8f0; padding: 30px; }}
        .container {{ max-width: 1100px; margin: 0 auto; }}
        h1 {{ color: #63b3ed; border-bottom: 2px solid #4a5568; padding-bottom: 10px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🩺 Severe DR Explainability & Inference Audit (Grade 3 Severe & Grade 4 Proliferative)</h1>
        <p>Testing ensemble performance and CLAHE + Spotlight contour localization on high-severity Diabetic Retinopathy fundus images.</p>
        {"".join(cards)}
    </div>
</body>
</html>
"""
    with html_file.open("w", encoding="utf-8") as f:
        f.write(gallery_html)

    print(f"  [OK] Severe cases gallery saved to: {html_file}")
    print("\n" + "=" * 75)
    print("TESTING COMPLETED SUCCESSFULLY!")
    print("=" * 75)


if __name__ == "__main__":
    main()
