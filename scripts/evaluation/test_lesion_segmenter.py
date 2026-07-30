"""Demonstration of Clinical Lesion Segmentation & Marker for Retinal Fundus Images.

Generates clinical medical annotations:
  - 🔴 Red circles: Hemorrhages & Microaneurysms (HE/MA)
  - 🟡 Yellow circles: Hard Exudates (EX)
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
from explainability.spotlight import clahe_enhance
from inference.predictor import RetinalPredictor

# ── Setup Paths ────────────────────────────────────────────────────────────────
CONFIG_PATH = Path("configs/experiments/exp_05_convnext_tiny_v2.yaml")
CKPT_PATH = Path("experiments/exp_05_convnext_tiny_v2/checkpoints/best.pt")
DATA_DIR = Path("data/raw/test")
OUTPUT_DIR = Path("reports/figures/clinical_lesion_segmentation")

TEST_IMAGES = [
    ("007-2809-100.jpg", 1, "Mild NPDR (Grade 1)"),
    ("007-5953-300.jpg", 2, "Moderate NPDR (Grade 2)"),
    ("007-5999-300.jpg", 3, "Severe NPDR (Grade 3)"),
    ("007-6962-400.jpg", 4, "Proliferative DR (Grade 4)"),
]


def main() -> None:
    print("=" * 80)
    print("CLINICAL MEDICAL LESION SEGMENTATION & ANNOTATION DEMO")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"-> Compute Device: {device}\n")

    print("[1/2] Initializing Predictor, Engine & Lesion Segmenter...")
    cfg = ConfigLoader.load(CONFIG_PATH)
    predictor = RetinalPredictor.from_checkpoint(
        checkpoint_path=CKPT_PATH, config=cfg, device=device
    )
    engine = ExplainabilityEngine(model=predictor.model)
    segmenter = LesionSegmenter()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_results = []

    print("[2/2] Generating Clinical Annotations for Severe Cases...\n")
    for img_name, gt_label, desc in TEST_IMAGES:
        img_path = DATA_DIR / img_name
        if not img_path.exists():
            print(f"  [SKIP] Image not found: {img_path}")
            continue

        print(
            f"--------------------------------------------------------------------------------"
        )
        print(f"File         : {img_name} ({desc})")

        bgr = cv2.imread(str(img_path))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        tensor = predictor._prepare_image_tensor(img_path)

        res = predictor.predict(img_path)
        print(
            f"Prediction   : Grade {res.grade_id} ({res.grade_name}) [Conf: {res.confidence:.2%}]"
        )

        # 1. Detect & Annotate Lesions
        annotated_rgb, counts = segmenter.annotate(
            rgb, draw_exudates=True, draw_hemorrhages=True
        )
        print(
            f"Detected     : [HE] Hemorrhages: {counts['hemorrhages']} | [EX] Hard Exudates: {counts['exudates']}"
        )

        # 2. Compute Grad-CAM++ heatmap
        heatmap = engine.generate_heatmap(
            input_tensor=tensor, algorithm="gradcam++", target_class=res.grade_id
        )

        # 3. Create 4-panel Clinical Diagnostic Report
        # Panel A: Original Image
        pw, ph = 360, 360
        panel_a = cv2.resize(rgb, (pw, ph))
        bar_a = np.zeros((26, pw, 3), dtype=np.uint8)
        cv2.putText(
            bar_a,
            "1. Original Fundus Image",
            (6, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )
        panel_a_labeled = np.vstack([panel_a, bar_a])

        # Panel B: CLAHE Contrast Enhanced
        clahe_img = clahe_enhance(rgb)
        panel_b = cv2.resize(clahe_img, (pw, ph))
        bar_b = np.zeros((26, pw, 3), dtype=np.uint8)
        cv2.putText(
            bar_b,
            "2. CLAHE Contrast Enhanced",
            (6, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )
        panel_b_labeled = np.vstack([panel_b, bar_b])

        # Panel C: Clinical Lesion Segmentation (HE/EX)
        panel_c_resized = cv2.resize(annotated_rgb, (pw, ph))
        bar_c = np.zeros((26, pw, 3), dtype=np.uint8)
        cv2.putText(
            bar_c,
            f"3. Lesion Marker (HE: {counts['hemorrhages']}, EX: {counts['exudates']})",
            (6, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )
        panel_c_labeled = np.vstack([panel_c_resized, bar_c])

        # Panel D: Grad-CAM++ Attention Heatmap
        h, w = rgb.shape[:2]
        hm_resized = cv2.resize(heatmap, (w, h))
        hm_uint8 = (np.clip(hm_resized, 0, 1) * 255).astype(np.uint8)
        colored = cv2.applyColorMap(hm_uint8, cv2.COLORMAP_JET)
        colored_rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
        blend = cv2.addWeighted(rgb, 0.55, colored_rgb, 0.45, 0)
        panel_d = cv2.resize(blend, (pw, ph))
        bar_d = np.zeros((26, pw, 3), dtype=np.uint8)
        cv2.putText(
            bar_d,
            "4. Grad-CAM++ AI Attention Map",
            (6, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )
        panel_d_labeled = np.vstack([panel_d, bar_d])

        # Combine 2x2 grid
        row_top = np.hstack([panel_a_labeled, panel_b_labeled])
        row_bot = np.hstack([panel_c_labeled, panel_d_labeled])

        # Top Title Header
        title_bar = np.zeros((32, pw * 2, 3), dtype=np.uint8)
        title_bar[:] = (25, 25, 25)
        title_txt = f"Clinical Report: {img_name} | GT: Grade {gt_label} | Pred: Grade {res.grade_id} ({res.grade_name}) [{res.confidence:.1%}]"
        cv2.putText(
            title_bar,
            title_txt,
            (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )

        report_4panel = np.vstack([title_bar, row_top, row_bot])
        out_png = OUTPUT_DIR / f"clinical_report_{img_path.stem}.png"
        cv2.imwrite(str(out_png), cv2.cvtColor(report_4panel, cv2.COLOR_RGB2BGR))
        print(f"Saved Clinical Report -> {out_png.name}\n")

        summary_results.append(
            {
                "image_id": img_name,
                "gt_label": gt_label,
                "pred_label": res.grade_id,
                "pred_name": res.grade_name,
                "confidence": res.confidence,
                "he_count": counts["hemorrhages"],
                "ex_count": counts["exudates"],
                "report_path": str(out_png),
            }
        )

    # Export HTML Report
    html_file = OUTPUT_DIR / "clinical_lesions_gallery.html"
    cards = []
    for r in summary_results:
        rel_path = Path(r["report_path"]).name
        cards.append(f"""
        <div style="background: #1a202c; border: 1px solid #2d3748; border-radius: 10px; overflow: hidden; margin-bottom: 30px;">
            <div style="padding: 15px; background: #2d3748; color: #90cdf4; display: flex; justify-content: space-between; align-items: center;">
                <h3 style="margin: 0;">{r["image_id"]} &mdash; Pred: Grade {r["pred_label"]} ({r["pred_name"]})</h3>
                <span style="background: #4a5568; color: #fff; padding: 4px 12px; border-radius: 4px;">🔴 HE: {r["he_count"]} | 🟡 EX: {r["ex_count"]}</span>
            </div>
            <img src="{rel_path}" style="width: 100%; display: block;" alt="Clinical Report">
        </div>
        """)

    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Clinical Lesion Segmentation & Marker Reports</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #0f1117; color: #e2e8f0; padding: 30px; }}
        .container {{ max-width: 1100px; margin: 0 auto; }}
        h1 {{ color: #63b3ed; border-bottom: 2px solid #4a5568; padding-bottom: 10px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🩺 Clinical Medical Lesion Marker & Diagnostic Reports</h1>
        <p>Medical-grade lesion detection: 🔴 Red circles = Hemorrhages (HE/MA) | 🟡 Yellow circles = Hard Exudates (EX)</p>
        {"".join(cards)}
    </div>
</body>
</html>
"""
    with html_file.open("w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"  [OK] HTML Clinical Lesions Gallery saved to: {html_file}")
    print("\n" + "=" * 80)
    print("CLINICAL LESION SEGMENTATION DEMO COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
