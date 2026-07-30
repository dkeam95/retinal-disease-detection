"""Official DDR Lesion Ground-Truth Validation & Comparison Script.

Compares official expert doctor lesion masks from `data/raw/lesion_segmentation/`
against the LesionSegmenter predictions and Grad-CAM++ AI attention maps.
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
from inference.predictor import RetinalPredictor

# ── Paths ──────────────────────────────────────────────────────────────────────
CONFIG_PATH = Path("configs/experiments/exp_05_convnext_tiny_v2.yaml")
CKPT_PATH = Path("experiments/exp_05_convnext_tiny_v2/checkpoints/best.pt")
DATA_DIR = Path("data/raw/test")
MASK_DIR = Path("data/raw/lesion_segmentation/test")
OUTPUT_DIR = Path("reports/figures/doctor_gt_comparison")


def load_official_doctor_mask(
    image_stem: str, height: int, width: int
) -> tuple[np.ndarray, dict[str, int]]:
    """Load and merge official doctor TIF masks for EX, HE, MA, SE."""
    merged_mask_bgr = np.zeros((height, width, 3), dtype=np.uint8)
    counts = {"ex": 0, "he": 0, "ma": 0, "se": 0}

    label_dir = MASK_DIR / "label"
    if not label_dir.exists():
        return merged_mask_bgr, counts

    # EX (Hard Exudates) -> Yellow (BGR 0, 240, 255)
    ex_file = label_dir / "EX" / f"{image_stem}.tif"
    if ex_file.exists():
        m_ex = cv2.imread(str(ex_file), cv2.IMREAD_GRAYSCALE)
        if m_ex is not None and np.any(m_ex > 0):
            m_ex_res = cv2.resize(
                m_ex, (width, height), interpolation=cv2.INTER_NEAREST
            )
            merged_mask_bgr[m_ex_res > 0] = (0, 240, 255)
            counts["ex"] = int(np.count_nonzero(m_ex_res))

    # HE (Hemorrhages) -> Red (BGR 0, 0, 255)
    he_file = label_dir / "HE" / f"{image_stem}.tif"
    if he_file.exists():
        m_he = cv2.imread(str(he_file), cv2.IMREAD_GRAYSCALE)
        if m_he is not None and np.any(m_he > 0):
            m_he_res = cv2.resize(
                m_he, (width, height), interpolation=cv2.INTER_NEAREST
            )
            merged_mask_bgr[m_he_res > 0] = (0, 0, 255)
            counts["he"] = int(np.count_nonzero(m_he_res))

    # MA (Microaneurysms) -> Magenta (BGR 255, 0, 255)
    ma_file = label_dir / "MA" / f"{image_stem}.tif"
    if ma_file.exists():
        m_ma = cv2.imread(str(ma_file), cv2.IMREAD_GRAYSCALE)
        if m_ma is not None and np.any(m_ma > 0):
            m_ma_res = cv2.resize(
                m_ma, (width, height), interpolation=cv2.INTER_NEAREST
            )
            merged_mask_bgr[m_ma_res > 0] = (255, 0, 255)
            counts["ma"] = int(np.count_nonzero(m_ma_res))

    return merged_mask_bgr, counts


def main() -> None:
    print("=" * 80)
    print("OFFICIAL DDR DOCTOR GROUND-TRUTH LESION VALIDATION & COMPARISON")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"-> Compute Device: {device}\n")

    # Find images that have active GT doctor masks
    label_dir = MASK_DIR / "label"
    candidate_stems = []

    for category in ["EX", "HE"]:
        cat_dir = label_dir / category
        if cat_dir.exists():
            for tif in cat_dir.glob("*.tif"):
                m = cv2.imread(str(tif), cv2.IMREAD_GRAYSCALE)
                if m is not None and np.count_nonzero(m) > 10:
                    candidate_stems.append(tif.stem)
                    if len(candidate_stems) >= 4:
                        break

    candidate_stems = sorted(list(set(candidate_stems)))[:4]
    print(
        f"[1/3] Found {len(candidate_stems)} target test images with official doctor GT masks: {candidate_stems}\n"
    )

    # Load Predictor, Engine & Segmenter
    cfg = ConfigLoader.load(CONFIG_PATH)
    predictor = RetinalPredictor.from_checkpoint(
        checkpoint_path=CKPT_PATH, config=cfg, device=device
    )
    engine = ExplainabilityEngine(model=predictor.model)
    segmenter = LesionSegmenter()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    print("[2/3] Generating Official Doctor GT vs AI Comparison Reports...\n")
    for stem in candidate_stems:
        # Search for corresponding JPG file in data/raw/test or data/raw/lesion_segmentation/test/image
        jpg_path = DATA_DIR / f"{stem}.jpg"
        if not jpg_path.exists():
            jpg_path = MASK_DIR / "image" / f"{stem}.jpg"
        if not jpg_path.exists():
            print(f"  [SKIP] Image file for {stem} not found.")
            continue

        bgr = cv2.imread(str(jpg_path))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]

        tensor = predictor._prepare_image_tensor(jpg_path)
        res = predictor.predict(jpg_path)

        # 1. Load official Doctor GT mask
        gt_mask_bgr, gt_counts = load_official_doctor_mask(stem, h, w)
        gt_mask_rgb = cv2.cvtColor(gt_mask_bgr, cv2.COLOR_BGR2RGB)
        gt_blend = cv2.addWeighted(rgb, 0.55, gt_mask_rgb, 0.45, 0)

        # 2. Algorithmic Lesion Marker
        annotated_rgb, algo_counts = segmenter.annotate(rgb)

        # 3. Grad-CAM++ Heatmap
        heatmap = engine.generate_heatmap(
            input_tensor=tensor,
            algorithm="gradcam++",
            target_class=res.grade_id,
            original_rgb=rgb,
        )

        # 4. Create 4-panel Doctor GT Comparison Report
        pw, ph = 360, 360
        # Panel A: Original Image
        p_a = cv2.resize(rgb, (pw, ph))
        b_a = np.zeros((26, pw, 3), dtype=np.uint8)
        cv2.putText(
            b_a,
            "1. Original Fundus Image",
            (6, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )
        p_a_lbl = np.vstack([p_a, b_a])

        # Panel B: Official Doctor Ground-Truth Lesion Mask
        p_b = cv2.resize(gt_blend, (pw, ph))
        b_b = np.zeros((26, pw, 3), dtype=np.uint8)
        cv2.putText(
            b_b,
            "2. Official Doctor GT Mask (EX/HE)",
            (6, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )
        p_b_lbl = np.vstack([p_b, b_b])

        # Panel C: Algorithmic Lesion Marker (LesionSegmenter)
        p_c = cv2.resize(annotated_rgb, (pw, ph))
        b_c = np.zeros((26, pw, 3), dtype=np.uint8)
        cv2.putText(
            b_c,
            f"3. Lesion Marker (HE: {algo_counts['hemorrhages']}, EX: {algo_counts['exudates']})",
            (6, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )
        p_c_lbl = np.vstack([p_c, b_c])

        # Panel D: Grad-CAM++ AI Attention Map
        hm_res = cv2.resize(heatmap, (w, h))
        hm_u8 = (np.clip(hm_res, 0, 1) * 255).astype(np.uint8)
        colored_hm = cv2.cvtColor(
            cv2.applyColorMap(hm_u8, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB
        )
        p_d_blend = cv2.addWeighted(rgb, 0.55, colored_hm, 0.45, 0)
        p_d = cv2.resize(p_d_blend, (pw, ph))
        b_d = np.zeros((26, pw, 3), dtype=np.uint8)
        cv2.putText(
            b_d,
            "4. Grad-CAM++ AI Attention Map",
            (6, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )
        p_d_lbl = np.vstack([p_d, b_d])

        # Grid Assembly
        r_top = np.hstack([p_a_lbl, p_b_lbl])
        r_bot = np.hstack([p_c_lbl, p_d_lbl])

        title = np.zeros((32, pw * 2, 3), dtype=np.uint8)
        title[:] = (25, 25, 25)
        txt = f"Doctor GT Validation: {stem}.jpg | Pred: Grade {res.grade_id} ({res.grade_name}) [{res.confidence:.1%}]"
        cv2.putText(
            title,
            txt,
            (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )

        report = np.vstack([title, r_top, r_bot])
        out_png = OUTPUT_DIR / f"doctor_gt_comparison_{stem}.png"
        cv2.imwrite(str(out_png), cv2.cvtColor(report, cv2.COLOR_RGB2BGR))
        print(f"  [OK] Saved Doctor GT Comparison Report -> {out_png.name}")

        results.append(
            {
                "image_id": f"{stem}.jpg",
                "pred_label": res.grade_id,
                "pred_name": res.grade_name,
                "confidence": res.confidence,
                "report_path": str(out_png),
            }
        )

    # Export HTML Report
    print("\n[3/3] Exporting HTML Official Doctor GT Validation Gallery...")
    html_path = OUTPUT_DIR / "doctor_gt_validation_report.html"
    cards = []
    for r in results:
        fname = Path(r["report_path"]).name
        cards.append(f"""
        <div style="background: #1a202c; border: 1px solid #2d3748; border-radius: 10px; overflow: hidden; margin-bottom: 30px;">
            <div style="padding: 15px; background: #2d3748; color: #63b3ed; font-weight: bold;">
                {r["image_id"]} &mdash; Pred: Grade {r["pred_label"]} ({r["pred_name"]}) [{r["confidence"]:.1%}]
            </div>
            <img src="{fname}" style="width: 100%; display: block;" alt="Doctor GT Comparison">
        </div>
        """)

    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Official DDR Doctor Ground-Truth Lesion Validation</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #0f1117; color: #e2e8f0; padding: 30px; }}
        .container {{ max-width: 1150px; margin: 0 auto; }}
        h1 {{ color: #63b3ed; border-bottom: 2px solid #4a5568; padding-bottom: 10px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🏥 Official DDR Doctor Ground-Truth Lesion Validation</h1>
        <p>Validation of AI Attention Maps and Lesion Markers against Official Expert Doctor TIF Masks from the DDR Dataset.</p>
        {"".join(cards)}
    </div>
</body>
</html>
"""
    with html_path.open("w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"  [OK] HTML Doctor GT Report saved to: {html_path}")
    print("\n" + "=" * 80)
    print("OFFICIAL DOCTOR GT VALIDATION COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
