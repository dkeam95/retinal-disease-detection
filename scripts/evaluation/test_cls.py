# ruff: noqa: E402
import os

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

import argparse
import random
import sys
from pathlib import Path

import torch

# 1. Add src/ directory to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.config.loader import ConfigLoader
from inference.predictor import RetinalPredictor


def load_annotations(
    data_dir: Path, ignore_classes: set[int] | None = None
) -> dict[str, int]:
    """Parses test.txt, valid.txt, train.txt into {filename: label}, ignoring unassessable classes."""
    if ignore_classes is None:
        ignore_classes = {5}
    annotations = {}
    txt_files = ["test.txt", "valid.txt", "train.txt"]

    for txt_name in txt_files:
        txt_path = data_dir / txt_name
        if not txt_path.exists():
            found = list(data_dir.rglob(txt_name))
            if found:
                txt_path = found[0]
            else:
                continue

        with open(txt_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) >= 2:
                    img_name = Path(parts[0]).name
                    try:
                        label = int(parts[1])
                        # Ignore unassessable/poor quality images (e.g. Grade 5)
                        if label in ignore_classes:
                            continue
                        annotations[img_name] = label
                    except ValueError:
                        continue
    return annotations


def select_file_via_gui(initial_dir: Path) -> Path | None:
    """Opens a native OS file dialog to select an image."""
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError:
        print("[WARN] Tkinter is not installed or available in this environment. Cannot open GUI.")
        return None

    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)

        file_path = filedialog.askopenfilename(
            title="Select Retinal Image for Inference",
            initialdir=str(initial_dir.resolve())
            if initial_dir.exists()
            else str(PROJECT_ROOT),
            filetypes=[
                ("Image Files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"),
                ("All Files", "*.*"),
            ],
        )
        root.destroy()
        return Path(file_path) if file_path else None
    except Exception as e:
        print(f"[WARN] Failed to initialize Tkinter GUI: {e}")
        return None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test classification predictor with Ground Truth validation (ignoring Class 5)."
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path or filename (optional). If omitted, File Explorer opens.",
    )
    return parser.parse_args()


def test_inference():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"-> Compute Device: {device}")

    # =========================================================================
    # UPDATED PATHS FOR YOUR NEW 512x512 EXPERIMENT
    # =========================================================================
    config_path = PROJECT_ROOT / "configs/experiments/exp_07_convnext_tiny_v4.yaml"
    checkpoint_dir = PROJECT_ROOT / "experiments/exp_07_convnext_tiny_v4/checkpoints"

    possible_checkpoints = [
        checkpoint_dir / "best_model.pt",
        checkpoint_dir / "best.pt",
        checkpoint_dir / "best_model.ckpt",
    ]

    checkpoint_path = None
    for ckpt in possible_checkpoints:
        if ckpt.exists():
            checkpoint_path = ckpt
            break

    if checkpoint_path is None or not checkpoint_path.exists():
        print(f"[ERROR] Checkpoint not found in directory: {checkpoint_dir}")
        return

    # 2. Load classification annotations (Class 5 is ignored)
    data_dir = PROJECT_ROOT / "data"
    gt_map = load_annotations(data_dir, ignore_classes={5})
    print(
        f"-> Loaded Ground Truth annotations for {len(gt_map)} valid images (Class 5 filtered out)."
    )

    # 3. Load predictor
    print("[1/2] Loading classification config & checkpoint...")
    cfg = ConfigLoader.load(config_path)
    predictor = RetinalPredictor.from_checkpoint(
        checkpoint_path=checkpoint_path, config=cfg, device=device
    )
    print(" [OK] Classification model loaded successfully!\n")

    default_test_dir = (
        PROJECT_ROOT / "data/raw/test"
        if (PROJECT_ROOT / "data/raw/test").exists()
        else PROJECT_ROOT / "data/raw"
    )
    image_path = None

    # CLI flag input
    if args.image:
        user_input = args.image.strip()
        input_path = Path(user_input)
        if input_path.exists():
            image_path = input_path
        else:
            matches = list(default_test_dir.rglob(f"*{user_input}*"))
            if matches:
                image_path = matches[0]

    # File Explorer GUI
    if image_path is None:
        print("[INFO] Opening File Explorer dialog...")
        image_path = select_file_via_gui(initial_dir=default_test_dir)

    # Fallback random image
    if image_path is None or not image_path.exists():
        print(
            "[WARN] No file selected via Explorer. Falling back to a random valid test image..."
        )
        found_images = (
            list(default_test_dir.rglob("*.jpg"))
            + list(default_test_dir.rglob("*.png"))
            + list(default_test_dir.rglob("*.jpeg"))
        )
        valid_candidates = [
            img
            for img in found_images
            if "label" not in str(img).lower()
            and "mask" not in str(img).lower()
            and (img.name in gt_map)
        ]

        if not valid_candidates:
            valid_candidates = found_images

        image_path = random.choice(valid_candidates)

    # 4. Run inference
    print(f"[2/2] Running inference on image: {image_path.name}")
    print(f"      Full Path: {image_path}")

    result = predictor.predict(image_path)

    # 5. Check Ground Truth status
    gt_label = gt_map.get(image_path.name, None)

    if gt_label is not None:
        match_status = (
            " MATCH [OK]"
            if gt_label == result.grade_id
            else " MISMATCH [FAILED]"
        )
        gt_display = f"Grade {gt_label}"
    else:
        match_status = " IGNORED / UNKNOWN"
        gt_display = "Grade 5 (Poor Quality - Skipped) or Not Found"

    print("\n" + "=" * 50)
    print("CLASSIFICATION & EVALUATION RESULTS:")
    print(f"  - Image Name   : {image_path.name}")
    print(f"  - Ground Truth : {gt_display}")
    print(f"  - Predicted    : Grade {result.grade_id} ({result.grade_name})")
    print(f"  - Confidence   : {result.confidence:.2%}")
    print(f"  - Validation   :{match_status}")
    print("=" * 50)


if __name__ == "__main__":
    test_inference()
