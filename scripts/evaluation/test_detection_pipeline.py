"""Demo / Test Script for Diabetic Retinopathy Lesion Detection (Faster R-CNN)."""

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
import torch

from common.config.loader import ConfigLoader
from inference.detection_predictor import LesionDetectionPredictor

CONFIG_PATH = Path("configs/detection/exp_faster_rcnn_lesions_v3_1536.yaml")
CKPT_PATH = Path("experiments/exp_faster_rcnn_lesions_v3_1536/checkpoints/best.pt")
TEST_XML_DIR = Path("data/raw/lesion_detection/test")
OUTPUT_DIR = Path("reports/figures/detection_demo")


def main() -> None:
    print("=" * 80)
    print("DIABETIC RETINOPATHY LESION DETECTION (FASTER R-CNN) DEMO")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"-> Compute Device: {device}\n")

    print("[1/2] Building Faster R-CNN (ResNet50-FPN) Detector with COCO pre-trained weights...")
    cfg = ConfigLoader.load(CONFIG_PATH)

    if CKPT_PATH.exists():
        print(f"  [OK] Found trained checkpoint -> {CKPT_PATH}")
        predictor = LesionDetectionPredictor.from_checkpoint(
            checkpoint_path=CKPT_PATH, config=cfg, device=device
        )
    else:
        print("  [INFO] No trained checkpoint found. Using initialized COCO pretrained model...")
        from detection.detector_model import build_lesion_detector

        model = build_lesion_detector(config=cfg.detection_model)
        predictor = LesionDetectionPredictor(
            model=model,
            device=device,
            score_thresh=cfg.detection_model.score_thresh,
            target_size=(cfg.detection_model.min_size, cfg.detection_model.max_size),
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Gather test image stems from XMLs
    test_stems = []
    if TEST_XML_DIR.exists():
        for xml_file in TEST_XML_DIR.glob("*.xml"):
            test_stems.append(xml_file.stem)
            if len(test_stems) >= 4:
                break

    if not test_stems:
        test_stems = ["007-1789-100", "007-1853-100", "007-2403-100", "007-2404-100"]

    results = []

    print("[2/2] Running Lesion Detection Inference & Rendering Overlays...\n")
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
            print(f"  [SKIP] Image for {stem} not found.")
            continue

        print(f"Target Image : {img_path.name}")
        bgr = cv2.imread(str(img_path))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        result = predictor.predict(img_path)
        print(f"  Detected   : {len(result.boxes)} lesion bounding boxes")

        counts = {}
        for name in result.class_names:
            counts[name] = counts.get(name, 0) + 1
        for cls_name, cnt in counts.items():
            print(f"    - {cls_name}: {cnt}")

        overlay_rgb = predictor.render_overlay(rgb, result, show_legend=True)

        out_name = f"lesion_detection_{stem}.png"
        out_path = OUTPUT_DIR / out_name
        cv2.imwrite(str(out_path), cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))
        print(f"  Saved      -> {out_name}\n")

        results.append(
            {
                "image_id": img_path.name,
                "boxes_count": len(result.boxes),
                "counts": counts,
                "report_path": str(out_path),
            }
        )

    # Export HTML Gallery
    html_file = OUTPUT_DIR / "lesion_detection_gallery.html"
    cards = []
    for r in results:
        fname = Path(r["report_path"]).name
        counts_str = " | ".join(
            [f"<strong>{k}</strong>: {v}" for k, v in r["counts"].items()]
        )
        cards.append(f"""
        <div style="background: #171923; border: 1px solid #2d3748; border-radius: 12px; padding: 20px; margin-bottom: 30px;">
            <h3 style="color: #63b3ed; margin-top: 0;">Sample: {r["image_id"]} (Total Lesions: {r["boxes_count"]})</h3>
            <p style="color: #cbd5e0; margin-bottom: 15px;">Breakdown: {counts_str}</p>
            <img src="{fname}" style="width: 100%; border-radius: 8px; border: 1px solid #4a5568;" alt="Detection Overlay">
        </div>
        """)

    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Diabetic Retinopathy Lesion Detection Gallery</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #0f1117; color: #e2e8f0; padding: 30px; }}
        .container {{ max-width: 1100px; margin: 0 auto; }}
        h1 {{ color: #63b3ed; border-bottom: 2px solid #4a5568; padding-bottom: 10px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔍 Diabetic Retinopathy Lesion Detection (Faster R-CNN)</h1>
        <p>Lesion bounding box detection fine-tuned on DDR PASCAL VOC XML annotations.</p>
        {"".join(cards)}
    </div>
</body>
</html>
"""
    with html_file.open("w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"  [OK] HTML Lesion Detection Gallery saved to: {html_file}")
    print("\n" + "=" * 80)
    print("LESION DETECTION DEMO COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
