"""Comprehensive Side-by-Side Benchmark of 5 SOTA XAI Algorithms on Retinal Fundus Images.

Algorithms Evaluated:
  1. Baseline Grad-CAM
  2. Grad-CAM++
  3. Layer-CAM
  4. Score-CAM
  5. Integrated Gradients
"""

from __future__ import annotations

import os

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

from pathlib import Path
import cv2
import numpy as np
import torch

from common.config.loader import ConfigLoader
from explainability.engine import ExplainabilityEngine
from explainability.spotlight import clahe_enhance
from inference.ensemble import EnsemblePredictor
from inference.predictor import RetinalPredictor

# ── Paths ──────────────────────────────────────────────────────────────────────
CONFIG_PATH = Path("configs/experiments/exp_05_convnext_tiny_v2.yaml")
CKPT_PATH = Path("experiments/exp_05_convnext_tiny_v2/checkpoints/best.pt")
DATA_DIR = Path("data/raw/test")
OUTPUT_DIR = Path("reports/figures/xai_benchmark")

TEST_IMAGES = [
    ("007-5999-300.jpg", 3, "Severe NPDR (Grade 3)"),
    ("007-6962-400.jpg", 4, "Proliferative DR (Grade 4)"),
]

ALGORITHMS = [
    ("gradcam", "Baseline Grad-CAM", "Coarse 7x7 regional activation"),
    ("gradcam++", "Grad-CAM++", "Higher order gradients for multiple lesions"),
    ("layer_cam", "Layer-CAM", "Fine-grained spatial gradient preservation"),
    ("score_cam", "Score-CAM", "Gradient-free perturbation confidence scoring"),
    (
        "integrated_gradients",
        "Integrated Gradients",
        "Pixel-level axiomatic attribution",
    ),
]


def main() -> None:
    print("=" * 80)
    print("COMPREHENSIVE BENCHMARK: 5 SOTA XAI ALGORITHMS ON RETINAL FUNDUS IMAGES")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"-> Compute Device: {device}\n")

    # Load Predictor & Engine
    print("[1/3] Loading ConvNeXt-Tiny Predictor & Explainability Engine...")
    cfg = ConfigLoader.load(CONFIG_PATH)
    predictor = RetinalPredictor.from_checkpoint(
        checkpoint_path=CKPT_PATH, config=cfg, device=device
    )
    engine = ExplainabilityEngine(model=predictor.model)
    print(
        "  [OK] Engine ready with 5 algorithms: "
        + ", ".join(engine.SUPPORTED_ALGORITHMS)
        + "\n"
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []

    # Process each test image
    print("[2/3] Generating Visual Attributions across All 5 Algorithms...\n")
    for img_name, gt_label, desc in TEST_IMAGES:
        img_path = DATA_DIR / img_name
        if not img_path.exists():
            print(f"  [SKIP] Image not found: {img_path}")
            continue

        print(
            f"--------------------------------------------------------------------------------"
        )
        print(f"Target Image : {img_name} ({desc})")

        bgr = cv2.imread(str(img_path))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        tensor = predictor._prepare_image_tensor(img_path)

        res = predictor.predict(img_path)
        print(
            f"Prediction   : Grade {res.grade_id} ({res.grade_name}) [Conf: {res.confidence:.2%}]"
        )

        image_benchmark = {
            "image_id": img_name,
            "gt_label": gt_label,
            "pred_label": res.grade_id,
            "pred_name": res.grade_name,
            "confidence": res.confidence,
            "algo_maps": [],
        }

        # Compute heatmaps for each of the 5 algorithms
        pw, ph = 280, 280
        comp_panels = []

        # Panel 0: Original + CLAHE
        clahe_img = clahe_enhance(rgb)
        clahe_resized = cv2.resize(clahe_img, (pw, ph))
        banner0 = np.zeros((24, pw, 3), dtype=np.uint8)
        cv2.putText(
            banner0,
            "Original (CLAHE)",
            (6, 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )
        comp_panels.append(np.vstack([clahe_resized, banner0]))

        for algo_key, algo_title, algo_desc in ALGORITHMS:
            heatmap = engine.generate_heatmap(
                input_tensor=tensor, algorithm=algo_key, target_class=res.grade_id
            )

            # Generate 4-panel report PNG
            png_filename = f"{img_path.stem}_{algo_key}_4panel.png"
            png_path = OUTPUT_DIR / png_filename
            engine.save_visualization(
                input_tensor=tensor,
                original_rgb=rgb,
                save_path=png_path,
                algorithm=algo_key,
                target_class=res.grade_id,
                grade_id=res.grade_id,
                class_name=res.grade_name,
                confidence=res.confidence,
                image_id=f"{img_path.stem} [{algo_title}]",
                mode="4panel",
            )

            # Generate single heatmap overlay tile for the side-by-side comparison bar
            hm_resized = cv2.resize(heatmap, (pw, ph))
            hm_uint8 = (np.clip(hm_resized, 0, 1) * 255).astype(np.uint8)
            colored = cv2.applyColorMap(hm_uint8, cv2.COLORMAP_JET)
            colored_rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
            blend = cv2.addWeighted(clahe_resized, 0.55, colored_rgb, 0.45, 0)

            banner = np.zeros((24, pw, 3), dtype=np.uint8)
            cv2.putText(
                banner,
                algo_title,
                (6, 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (220, 220, 220),
                1,
                cv2.LINE_AA,
            )
            comp_panels.append(np.vstack([blend, banner]))

            image_benchmark["algo_maps"].append(
                {
                    "algo_key": algo_key,
                    "title": algo_title,
                    "description": algo_desc,
                    "png_path": str(png_path),
                }
            )
            print(f"  [OK] Computed {algo_title:<22} -> {png_filename}")

        # Combine into single 1x6 comparison strip
        strip = np.hstack(comp_panels)
        strip_path = OUTPUT_DIR / f"comparison_strip_{img_path.stem}.png"
        cv2.imwrite(str(strip_path), cv2.cvtColor(strip, cv2.COLOR_RGB2BGR))
        image_benchmark["strip_path"] = str(strip_path)
        print(f"  [OK] Saved 1x6 Comparison Strip -> {strip_path.name}\n")

        all_results.append(image_benchmark)

    # 3. Export HTML Comparison Report
    print("[3/3] Exporting Interactive HTML Comparison Benchmark...")
    html_path = OUTPUT_DIR / "xai_benchmark_report.html"

    sections = []
    for res in all_results:
        strip_name = Path(res["strip_path"]).name
        cards = []
        for a in res["algo_maps"]:
            png_name = Path(a["png_path"]).name
            cards.append(f"""
            <div style="background: #1a202c; border: 1px solid #2d3748; border-radius: 8px; overflow: hidden; margin-bottom: 15px;">
                <div style="padding: 10px 15px; background: #2d3748; color: #63b3ed; font-weight: bold;">
                    {a["title"]} <span style="font-weight: normal; color: #a0aec0; font-size: 12px;">— {a["description"]}</span>
                </div>
                <img src="{png_name}" style="width: 100%; display: block;" alt="{a["title"]}">
            </div>
            """)

        sections.append(f"""
        <div style="margin-bottom: 40px; background: #171923; border: 1px solid #2d3748; border-radius: 12px; padding: 20px;">
            <h2 style="color: #ed8936; margin-top: 0;">Sample: {res["image_id"]} (GT Grade {res["gt_label"]} vs Pred Grade {res["pred_label"]})</h2>
            <p style="color: #cbd5e0;">Predicted: <strong>Grade {res["pred_label"]} ({res["pred_name"]})</strong> &nbsp;|&nbsp; Confidence: <strong>{res["confidence"]:.2%}</strong></p>
            
            <h4 style="color: #90cdf4; margin-top: 20px;">1x6 Side-by-Side Quick Comparison:</h4>
            <img src="{strip_name}" style="width: 100%; border-radius: 8px; border: 1px solid #4a5568;" alt="Comparison Strip">
            
            <h4 style="color: #90cdf4; margin-top: 25px;">Detailed 4-Panel Reports for Each Algorithm:</h4>
            {"".join(cards)}
        </div>
        """)

    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>SOTA XAI Benchmark Comparison</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #0f1117; color: #e2e8f0; padding: 30px; }}
        .container {{ max-width: 1300px; margin: 0 auto; }}
        h1 {{ color: #63b3ed; border-bottom: 2px solid #4a5568; padding-bottom: 10px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>👁️ Retinal Explainability Suite: Side-by-Side SOTA Benchmark</h1>
        <p>Comprehensive comparison of 5 Explainable AI algorithms (Grad-CAM, Grad-CAM++, Layer-CAM, Score-CAM, Integrated Gradients) on Diabetic Retinopathy fundus images.</p>
        {"".join(sections)}
    </div>
</body>
</html>
"""
    with html_path.open("w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"  [OK] HTML Benchmark Report saved to: {html_path}")
    print("\n" + "=" * 80)
    print("ALL 5 XAI ALGORITHMS EVALUATED & BENCHMARKED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
