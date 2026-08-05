# ruff: noqa: E402
"""Test script for 3-Model Ensemble Retinal Disease Classification (DR Severity Grading).

Loads the 3-model weighted ensemble (ConvNeXt-Tiny v4 + Swin-T + DenseNet-121)
and performs inference on a selected retinal fundus image, comparing predictions with Ground Truth labels.

Usage:
    # 1. Open GUI File Explorer to select an image:
    python scripts/evaluation/test_cls.py

    # 2. Process specific image by filename or path:
    python scripts/evaluation/test_cls.py --image data/raw/test/007-1811-100.jpg
    python scripts/evaluation/test_cls.py --image 007-1811-100.jpg
"""

import argparse
import os
import random
import sys
from pathlib import Path

import torch

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

# Add project root and src/ directory to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.config.loader import ConfigLoader
from inference.ensemble import EnsemblePredictor
from inference.predictor import RetinalPredictor

# ── 3-Model Ensemble Configuration ─────────────────────────────────────────────
ENSEMBLE_MODELS = [
    {
        "name": "ConvNeXt-Tiny v4",
        "config": PROJECT_ROOT / "configs/experiments/exp_07_convnext_tiny_v4.yaml",
        "checkpoint": PROJECT_ROOT / "experiments/exp_07_convnext_tiny_v4/checkpoints/best.pt",
        "weight": 0.40,
    },
    {
        "name": "Swin-Transformer T",
        "config": PROJECT_ROOT / "configs/experiments/exp_07_swin_t.yaml",
        "checkpoint": PROJECT_ROOT / "experiments/exp_07_swin_t/checkpoints/best.pt",
        "weight": 0.35,
    },
    {
        "name": "DenseNet-121",
        "config": PROJECT_ROOT / "configs/experiments/exp_02_densenet121.yaml",
        "checkpoint": PROJECT_ROOT / "experiments/exp_02_densenet121/checkpoints/best.pt",
        "weight": 0.25,
    },
]


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

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)

        file_path = filedialog.askopenfilename(
            title="Select Retinal Image for Ensemble Inference",
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
        description="Run 3-Model Ensemble Classification Inference on Retinal Fundus Images."
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path or filename of target retinal image. If omitted, File Explorer opens.",
    )
    return parser.parse_args()


def load_3model_ensemble(device: torch.device) -> tuple[EnsemblePredictor, list[dict]]:
    """Loads and initializes the 3-model weighted ensemble."""
    predictors: list[RetinalPredictor] = []
    weights: list[float] = []
    loaded_info: list[dict] = []

    print("[1/2] Loading 3-Model Ensemble checkpoints...")

    for model_info in ENSEMBLE_MODELS:
        name = model_info["name"]
        cfg_p = model_info["config"]
        ckpt_p = model_info["checkpoint"]

        if not cfg_p.exists() or not ckpt_p.exists():
            print(f"  [WARN] Skipping model '{name}': Config or Checkpoint missing.")
            continue

        print(f"  * Loading {name} (Weight: {model_info['weight']:.0%})...")
        cfg = ConfigLoader.load(cfg_p)
        predictor = RetinalPredictor.from_checkpoint(
            checkpoint_path=ckpt_p, config=cfg, device=device
        )
        predictors.append(predictor)
        weights.append(model_info["weight"])
        loaded_info.append({"name": name, "weight": model_info["weight"]})

    if not predictors:
        raise RuntimeError("Failed to load any models for EnsemblePredictor!")

    ensemble = EnsemblePredictor(predictors=predictors, weights=weights)
    print(f" [SUCCESS] Loaded {len(predictors)} model(s) into EnsemblePredictor.\n")
    return ensemble, loaded_info


def test_inference():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"-> Compute Device: {device}")

    # Load 3-Model Ensemble
    ensemble, loaded_models_info = load_3model_ensemble(device)

    # Load Ground Truth mapping
    data_dir = PROJECT_ROOT / "data"
    gt_map = load_annotations(data_dir, ignore_classes={5})
    print(f"-> Loaded Ground Truth annotations for {len(gt_map)} valid images.")

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
            matches = list(data_dir.rglob(f"*{user_input}*"))
            valid_m = [m for m in matches if m.is_file() and m.suffix.lower() in ('.jpg', '.png', '.jpeg')]
            if valid_m:
                image_path = valid_m[0]
            else:
                print(f"[ERROR] Could not find image matching '{user_input}' in data/.")
                sys.exit(1)

    # File Explorer GUI
    if image_path is None:
        print("[INFO] Opening File Explorer dialog...")
        image_path = select_file_via_gui(initial_dir=default_test_dir)

    # Fallback random image
    if image_path is None or not image_path.exists():
        print("[WARN] No file selected. Falling back to a random valid test image...")
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

    # 4. Run Ensemble Inference
    print(f"[2/2] Running 3-Model Ensemble Inference on image: {image_path.name}")
    print(f"      Full Path: {image_path}\n")

    # Individual model breakdown
    print("--- Individual Model Predictions ---")
    for info, pred in zip(loaded_models_info, ensemble.predictors, strict=False):
        ind_res = pred.predict(image_path)
        print(f"  * {info['name']:<24} (w={info['weight']:.2f}) -> Grade {ind_res.grade_id} ({ind_res.grade_name}) [{ind_res.confidence:.1%}]")

    # Ensemble combined prediction
    result = ensemble.predict(image_path)

    # Check Ground Truth
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

    print("\n" + "=" * 60)
    print(" [RESULT] 3-MODEL ENSEMBLE DIAGNOSTIC RESULTS")
    print("=" * 60)
    print(f"  - Image Name       : {image_path.name}")
    print(f"  - Ground Truth     : {gt_display}")
    print(f"  - Ensemble Grade   : Grade {result.grade_id} ({result.grade_name})")
    print(f"  - Combined Conf.   : {result.confidence:.2%}")
    print(f"  - Validation Status:{match_status}")
    print("  - Grade Distribution:")
    for g_id, (g_name, prob) in enumerate(zip(ensemble.class_names, result.probabilities, strict=False)):
        bar = "#" * int(prob * 30)
        print(f"      Grade {g_id} ({g_name:<18}): {prob:6.1%}  {bar}")
    print("=" * 60)


if __name__ == "__main__":
    test_inference()
