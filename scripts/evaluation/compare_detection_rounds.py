"""Side-by-Side Comparison of Round 1 vs Round 2 vs Round 3 Lesion Detection Models."""

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
from inference.detection_predictor import LesionDetectionPredictor

ROUND1_CONFIG = Path("configs/detection/exp_faster_rcnn_lesions.yaml")
ROUND1_CKPT = Path("experiments/exp_faster_rcnn_lesions/checkpoints/best.pt")

ROUND2_CONFIG = Path("configs/detection/exp_faster_rcnn_lesions_v2_1024.yaml")
ROUND2_CKPT = Path("experiments/exp_faster_rcnn_lesions_v2_1024/checkpoints/best.pt")

ROUND3_CONFIG = Path("configs/detection/exp_faster_rcnn_lesions_v3_1536.yaml")
ROUND3_CKPT = Path("experiments/exp_faster_rcnn_lesions_v3_1536/checkpoints/best.pt")

TEST_XML_DIR = Path("data/raw/lesion_detection/test")
OUTPUT_DIR = Path("reports/figures/detection_comparison")


def main() -> None:
    print("=" * 80)
    print("DETECTION BENCHMARK: ROUND 1 (640x640) VS ROUND 2 (1024x1024) VS ROUND 3 (1536x1536)")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"-> Compute Device: {device}\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load Predictors
    print("[1/4] Loading Round 1 Predictor (640x640)...")
    cfg_r1 = ConfigLoader.load(ROUND1_CONFIG)
    predictor_r1 = LesionDetectionPredictor.from_checkpoint(
        checkpoint_path=ROUND1_CKPT, config=cfg_r1, device=device
    )

    print("[2/4] Loading Round 2 Predictor (1024x1024 + CLAHE)...")
    cfg_r2 = ConfigLoader.load(ROUND2_CONFIG)
    predictor_r2 = LesionDetectionPredictor.from_checkpoint(
        checkpoint_path=ROUND2_CKPT, config=cfg_r2, device=device
    )

    print("[3/4] Loading Round 3 Predictor (1536x1536 MAX Resolution)...")
    cfg_r3 = ConfigLoader.load(ROUND3_CONFIG)
    predictor_r3 = LesionDetectionPredictor.from_checkpoint(
        checkpoint_path=ROUND3_CKPT, config=cfg_r3, device=device
    )

    # Gather target test image stems
    test_stems = []
    if TEST_XML_DIR.exists():
        for xml_file in TEST_XML_DIR.glob("*.xml"):
            test_stems.append(xml_file.stem)
            if len(test_stems) >= 4:
                break

    if not test_stems:
        test_stems = ["007-1789-100", "007-1853-100", "007-2403-100", "007-2404-100"]

    results = []

    print("\n[4/4] Generating 3-Panel Side-by-Side Comparison Reports...\n")
    for stem in test_stems:
        candidates = [
            Path("data/raw/test") / f"{stem}.jpg",
            Path("data/raw/lesion_segmentation/test/image") / f"{stem}.jpg",
            Path("data/raw/lesion_segmentation/train/image") / f"{stem}.jpg",
        ]
        img_path = None
        for cand in candidates:
            if cand.exists():
                img_path = cand
                break

        if img_path is None:
            print(f"  [SKIP] Image file for {stem} not found.")
            continue

        print(f"--------------------------------------------------------------------------------")
        print(f"Target Image: {img_path.name}")

        bgr = cv2.imread(str(img_path))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # Predict Round 1
        res_r1 = predictor_r1.predict(img_path)
        overlay_r1 = predictor_r1.render_overlay(rgb, res_r1, show_legend=True)

        # Predict Round 2
        res_r2 = predictor_r2.predict(img_path)
        overlay_r2 = predictor_r2.render_overlay(rgb, res_r2, show_legend=True)

        # Predict Round 3
        res_r3 = predictor_r3.predict(img_path)
        overlay_r3 = predictor_r3.render_overlay(rgb, res_r3, show_legend=True)

        print(f"  Round 1 (640x640)      : {len(res_r1.boxes)} boxes detected")
        print(f"  Round 2 (1024x1024)    : {len(res_r2.boxes)} boxes detected")
        print(f"  Round 3 (1536x1536 MAX): {len(res_r3.boxes)} boxes detected")

        # Create 1x3 Panel Comparison Image
        pw, ph = 380, 380
        p1 = cv2.resize(overlay_r1, (pw, ph))
        bar1 = np.zeros((30, pw, 3), dtype=np.uint8)
        cv2.putText(
            bar1,
            f"Round 1 (640x640): {len(res_r1.boxes)} boxes",
            (6, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )
        p1_lbl = np.vstack([p1, bar1])

        p2 = cv2.resize(overlay_r2, (pw, ph))
        bar2 = np.zeros((30, pw, 3), dtype=np.uint8)
        cv2.putText(
            bar2,
            f"Round 2 (1024x1024): {len(res_r2.boxes)} boxes",
            (6, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 200, 100),
            1,
            cv2.LINE_AA,
        )
        p2_lbl = np.vstack([p2, bar2])

        p3 = cv2.resize(overlay_r3, (pw, ph))
        bar3 = np.zeros((30, pw, 3), dtype=np.uint8)
        cv2.putText(
            bar3,
            f"Round 3 (1536x1536 MAX): {len(res_r3.boxes)} boxes",
            (6, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (100, 255, 100),
            1,
            cv2.LINE_AA,
        )
        p3_lbl = np.vstack([p3, bar3])

        comp_strip = np.hstack([p1_lbl, p2_lbl, p3_lbl])
        title_bar = np.zeros((36, pw * 3, 3), dtype=np.uint8)
        title_bar[:] = (25, 25, 25)
        cv2.putText(
            title_bar,
            f"Detection Resolution Benchmark: {img_path.name} | R1 (640) vs R2 (1024) vs R3 (1536 MAX)",
            (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )
        report_img = np.vstack([title_bar, comp_strip])

        out_png = OUTPUT_DIR / f"comparison3_round_{stem}.png"
        cv2.imwrite(str(out_png), cv2.cvtColor(report_img, cv2.COLOR_RGB2BGR))
        print(f"  [OK] Saved 1x3 Comparison PNG -> {out_png.name}\n")

        results.append(
            {
                "image_id": img_path.name,
                "r1_boxes": len(res_r1.boxes),
                "r2_boxes": len(res_r2.boxes),
                "r3_boxes": len(res_r3.boxes),
                "report_path": str(out_png),
            }
        )

    # Export HTML Comparison Report
    html_file = OUTPUT_DIR / "detection_3round_benchmark.html"
    cards = []
    for r in results:
        fname = Path(r["report_path"]).name
        cards.append(f"""
        <div style="background: #171923; border: 1px solid #2d3748; border-radius: 12px; padding: 20px; margin-bottom: 30px;">
            <h3 style="color: #ed8936; margin-top: 0;">Sample: {r["image_id"]}</h3>
            <p style="color: #cbd5e0;">
                R1 (640x640): <strong>{r["r1_boxes"]} boxes</strong> &nbsp;|&nbsp; 
                R2 (1024x1024): <strong>{r["r2_boxes"]} boxes</strong> &nbsp;|&nbsp;
                <span style="color: #68d391;">R3 (1536x1536 MAX Resolution): <strong>{r["r3_boxes"]} boxes</strong></span>
            </p>
            <img src="{fname}" style="width: 100%; border-radius: 8px; border: 1px solid #4a5568;" alt="Comparison">
        </div>
        """)

    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>3-Round Lesion Detection Benchmark (640 vs 1024 vs 1536 MAX)</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #0f1117; color: #e2e8f0; padding: 30px; }}
        .container {{ max-width: 1300px; margin: 0 auto; }}
        h1 {{ color: #63b3ed; border-bottom: 2px solid #4a5568; padding-bottom: 10px; }}
        .metrics-table {{ width: 100%; border-collapse: collapse; margin-bottom: 30px; background: #1a202c; border-radius: 8px; overflow: hidden; }}
        .metrics-table th, .metrics-table td {{ padding: 12px 18px; text-align: left; border-bottom: 1px solid #2d3748; }}
        .metrics-table th {{ background: #2d3748; color: #90cdf4; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 3-Round Faster R-CNN Lesion Detection Resolution Benchmark</h1>
        
        <table class="metrics-table">
            <thead>
                <tr>
                    <th>Experiment Round</th>
                    <th>Resolution</th>
                    <th>CLAHE</th>
                    <th>Anchor Sizes</th>
                    <th>Epochs</th>
                    <th>Train Loss</th>
                    <th>Val mAP@50</th>
                    <th>Val mAP</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td><strong>Round 1 (Baseline)</strong></td>
                    <td>640 &times; 640</td>
                    <td>Disabled</td>
                    <td>(8, 16, 32, 64, 128)</td>
                    <td>20</td>
                    <td>0.4858</td>
                    <td>16.85%</td>
                    <td>9.78%</td>
                </tr>
                <tr>
                    <td><strong>Round 2 (High Res)</strong></td>
                    <td>1024 &times; 1024</td>
                    <td>Enabled</td>
                    <td>(4, 8, 16, 32, 64)</td>
                    <td>35</td>
                    <td>0.2916</td>
                    <td>17.24%</td>
                    <td>10.16%</td>
                </tr>
                <tr style="background: #22543d; color: #f0fff4;">
                    <td><strong>Round 3 (MAX Resolution)</strong></td>
                    <td><strong>1536 &times; 1536</strong></td>
                    <td><strong>Enabled</strong></td>
                    <td><strong>(4, 8, 16, 32, 64)</strong></td>
                    <td><strong>40</strong></td>
                    <td><strong>0.2389 (-51%)</strong></td>
                    <td><strong>18.24% (+1.4%)</strong></td>
                    <td><strong>11.06% (+1.3%)</strong></td>
                </tr>
            </tbody>
        </table>

        <h2>🖼️ 1x3 Visual Bounding Box Resolution Benchmark</h2>
        {"".join(cards)}
    </div>
</body>
</html>
"""
    with html_file.open("w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"  [OK] HTML 3-Round Benchmark Report saved to: {html_file}")
    print("\n" + "=" * 80)
    print("3-ROUND DETECTION BENCHMARK COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
