# ruff: noqa: E402
"""Interactive & CLI Tool for Retinal Lesion Mask Overlay Visualization.

Allows selecting any test image (via CLI argument, interactive menu, or GUI file picker)
and renders precision medical color-coded mask overlays (EX, HE, MA, SE) and
multi-stage pipeline comparisons.

Usage:
    # 1. Interactive terminal selection menu:
    python scripts/visualization/visualize_mask_overlay.py

    # 2. Specific image by path or filename:
    python scripts/visualization/visualize_mask_overlay.py --image data/raw/DDR-dataset/lesion_segmentation/test/image/007-3396-200.jpg
    python scripts/visualization/visualize_mask_overlay.py --image 007-1774-100.jpg

    # 3. Native OS GUI file picker:
    python scripts/visualization/visualize_mask_overlay.py --gui
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

# Ensure project root and src/ are in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.config import ConfigLoader
from explainability.lesion_segmenter import LesionSegmenter
from inference.detection_predictor import LesionDetectionPredictor
from inference.masked_predictor import MaskedLesionPredictor
from visualization.mask_overlay import LesionMaskVisualizer


def find_checkpoint_path() -> Path | None:
    """Find available trained detection checkpoint weights."""
    candidates = [
        PROJECT_ROOT / "experiments" / "exp_faster_rcnn_lesions_v3_1536" / "checkpoints" / "best.pt",
        PROJECT_ROOT / "experiments" / "exp_faster_rcnn_lesions_v3_1280" / "checkpoints" / "best.pt",
        PROJECT_ROOT / "checkpoints" / "best.pt",
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    return None


def select_file_via_gui() -> Path | None:
    """Open native OS file selection dialog."""
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)

        initial_dir = PROJECT_ROOT / "data"
        if not initial_dir.exists():
            initial_dir = PROJECT_ROOT

        file_path = filedialog.askopenfilename(
            title="Select Retinal Fundus Image for Mask Overlay",
            initialdir=str(initial_dir.resolve()),
            filetypes=[
                ("Image Files", "*.jpg *.jpeg *.png *.tif *.tiff *.bmp"),
                ("All Files", "*.*"),
            ],
        )
        root.destroy()
        return Path(file_path) if file_path else None
    except Exception as e:
        print(f"[WARN] GUI file dialog unavailable: {e}")
        return None


def list_available_test_images(limit: int = 25) -> list[Path]:
    """Find test fundus images in data directory."""
    data_dir = PROJECT_ROOT / "data"
    if not data_dir.exists():
        return []

    # Priority directories for test images
    priority_subdirs = [
        "raw/DDR-dataset/lesion_segmentation/test/image",
        "raw/DDR-dataset/DR_grading/test",
        "raw/test",
        "test",
    ]

    found_files: list[Path] = []
    for p_sub in priority_subdirs:
        sub_p = data_dir / p_sub
        if sub_p.exists():
            for ext in ("*.jpg", "*.png", "*.jpeg", "*.tif"):
                found_files.extend(list(sub_p.glob(ext)))
                if len(found_files) >= limit:
                    break

    if not found_files:
        for ext in ("*.jpg", "*.png", "*.jpeg", "*.tif"):
            found_files.extend(list(data_dir.rglob(ext)))
            if len(found_files) >= limit:
                break

    # Sort files deterministically
    return sorted(list(set(found_files)))[:limit]


def prompt_user_for_image() -> Path | None:
    """Interactive CLI menu to pick an image from list or input custom path."""
    images = list_available_test_images()

    print("\n=======================================================")
    print(" SELECT RETINAL IMAGE FOR MASK OVERLAY VISUALIZATION")
    print("=======================================================\n")

    if images:
        print("Available test images found in data/:")
        for idx, img_p in enumerate(images, 1):
            rel_p = img_p.relative_to(PROJECT_ROOT) if img_p.is_relative_to(PROJECT_ROOT) else img_p
            print(f" [{idx:2d}] {img_p.name:<24} ({rel_p})")
        print("\n Options:")
        print(f"   * Enter number (1-{len(images)}) to select from list")
        print("   * Or paste full/relative path to any image")
        print("   * Or press [Enter] to use default test image\n")
    else:
        print("No image dataset auto-detected in data/. Please enter an image path:\n")

    try:
        user_input = input("Selection > ").strip()
    except (EOFError, KeyboardInterrupt):
        return None

    if not user_input:
        return images[0] if images else None

    if user_input.isdigit():
        choice_idx = int(user_input) - 1
        if 0 <= choice_idx < len(images):
            return images[choice_idx]
        else:
            print(f"[ERROR] Choice {user_input} is out of range.")
            return None

    # Search by filename substring or path
    user_p = Path(user_input)
    if user_p.exists() and user_p.is_file():
        return user_p

    # Try searching inside project root
    matches = list(PROJECT_ROOT.rglob(f"*{user_input}*"))
    valid_matches = [m for m in matches if m.is_file() and m.suffix.lower() in ('.jpg', '.png', '.jpeg', '.tif')]
    if valid_matches:
        print(f"[INFO] Found matching image: {valid_matches[0]}")
        return valid_matches[0]

    print(f"[ERROR] Image not found: {user_input}")
    return None


def run_visualization(
    image_path: Path,
    checkpoint_path: Path | None = None,
    output_dir: Path | None = None,
) -> None:
    """Generate and save mask overlays for the selected image."""
    if output_dir is None:
        output_dir = PROJECT_ROOT / "reports" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[INFO] Loading image: {image_path}")
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise ValueError(f"Failed to read image from: {image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Compute Device: {device}")

    visualizer = LesionMaskVisualizer(alpha=0.40, contour_thickness=2)
    output_files: list[tuple[str, Path]] = []

    # 1. Try ML-based MaskedLesionPredictor (Faster R-CNN)
    ckp_p = checkpoint_path or find_checkpoint_path()
    if ckp_p and ckp_p.exists():
        print(f"[INFO] Loading Faster R-CNN Checkpoint: {ckp_p.name}")
        try:
            cfg = ConfigLoader.load(PROJECT_ROOT / "configs" / "config.yaml")
            base_pred = LesionDetectionPredictor.from_checkpoint(
                ckp_p, config=cfg, device=device
            )
            base_pred.score_thresh = 0.25
            masked_pred = MaskedLesionPredictor(
                base_predictor=base_pred,
                use_fov_crop=True,
                refine_edges=True,
                mask_confidence_thresh=0.45,
            )

            result = masked_pred.predict_masked(rgb)
            print(f"[SUCCESS] Model detected {len(result.boxes)} lesion region(s).")
            for cls_n, sc in zip(result.class_names, result.scores, strict=False):
                print(f"   * {cls_n}: {sc:.1%}")

            if result.masks:
                # Color Overlay
                overlay_rgb = visualizer.draw_mask_overlay(
                    image_rgb=rgb,
                    masks=result.masks,
                    labels=result.labels,
                    draw_contours=True,
                    high_contrast_bw=False,
                )
                side_by_side = np.hstack([rgb, overlay_rgb])
                out_side_p = output_dir / f"mask_overlay_side_by_side_{image_path.stem}.png"
                cv2.imwrite(str(out_side_p), cv2.cvtColor(side_by_side, cv2.COLOR_RGB2BGR))
                output_files.append(("Side-by-Side Mask Overlay", out_side_p))

                # High contrast B&W background overlay
                bw_overlay = visualizer.draw_mask_overlay(
                    image_rgb=rgb,
                    masks=result.masks,
                    labels=result.labels,
                    draw_contours=True,
                    high_contrast_bw=True,
                )
                out_bw_p = output_dir / f"mask_overlay_high_contrast_{image_path.stem}.png"
                cv2.imwrite(str(out_bw_p), cv2.cvtColor(bw_overlay, cv2.COLOR_RGB2BGR))
                output_files.append(("High-Contrast B&W Mask Overlay", out_bw_p))

        except Exception as e:
            print(f"[WARN] Error running ML MaskedPredictor: {e}")

    # 2. Run Rule-Based Clinical Lesion Segmenter
    print("[INFO] Running Rule-Based Clinical Lesion Segmenter (EX & HE/MA)...")
    segmenter = LesionSegmenter()
    clinical_marked, counts = segmenter.annotate(rgb, draw_exudates=True, draw_hemorrhages=True, show_legend=True)

    print(f"   * Hard Exudates (EX) detected: {counts.get('exudates', 0)} blob(s)")
    print(f"   * Hemorrhages/Microaneurysms (HE/MA) detected: {counts.get('hemorrhages', 0)} blob(s)")

    out_clinical_p = output_dir / f"clinical_segmentation_marked_{image_path.stem}.png"
    cv2.imwrite(str(out_clinical_p), cv2.cvtColor(clinical_marked, cv2.COLOR_RGB2BGR))
    output_files.append(("Clinical Rule-Based Lesion Overlay", out_clinical_p))

    # Print Summary Results
    print("\n=======================================================")
    print(" [SUCCESS] VISUALIZATION COMPLETED SUCCESSFULLY!")
    print("=======================================================")
    print(f"Input image: {image_path.resolve()}\n")
    print("Generated Artifacts:")
    for desc, path in output_files:
        print(f"   * {desc}:")
        print(f"     File Path: {path.resolve()}")
        print(f"     Markdown Link: [{path.name}](file:///{path.resolve().as_posix()})")
    print("=======================================================\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive & CLI Tool for Retinal Lesion Mask Overlay Visualization."
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path or filename of target retinal image.",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Open native OS GUI file selector dialog.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to custom model weights (.pt).",
    )
    args = parser.parse_args()

    target_image: Path | None = None

    if args.gui:
        target_image = select_file_via_gui()
    elif args.image:
        p = Path(args.image)
        if p.exists() and p.is_file():
            target_image = p
        else:
            # Try searching project root
            matches = list(PROJECT_ROOT.rglob(f"*{args.image}*"))
            valid = [m for m in matches if m.is_file() and m.suffix.lower() in ('.jpg', '.png', '.jpeg')]
            if valid:
                target_image = valid[0]
            else:
                print(f"[ERROR] Image '{args.image}' not found.")
                sys.exit(1)
    else:
        target_image = prompt_user_for_image()

    if target_image is None:
        print("[CANCELLED] No image selected.")
        sys.exit(0)

    ckp_p = Path(args.checkpoint) if args.checkpoint else None
    run_visualization(target_image, checkpoint_path=ckp_p)


if __name__ == "__main__":
    main()
