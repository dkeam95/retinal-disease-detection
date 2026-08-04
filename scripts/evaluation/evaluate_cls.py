# ruff: noqa: E402
import os

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

import contextlib
import io
import sys
from pathlib import Path

import torch
from sklearn.metrics import classification_report, cohen_kappa_score, confusion_matrix
from tqdm import tqdm

# 1. Add src/ directory to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.config.loader import ConfigLoader
from inference.predictor import RetinalPredictor


def load_dataset_split(
    data_dir: Path, split_file: str = "test.txt", ignore_classes: set[int] | None = None
) -> list[tuple[Path, int]]:
    """Loads dataset split from txt annotation and filters out ignored classes (e.g. Grade 5)."""
    if ignore_classes is None:
        ignore_classes = {5}
    samples = []
    txt_path = data_dir / split_file

    if not txt_path.exists():
        found = list(data_dir.rglob(split_file))
        if found:
            txt_path = found[0]
        else:
            raise FileNotFoundError(
                f"[ERROR] Could not find annotation file: {split_file}"
            )

    # Possible directory paths for raw test images
    possible_image_dirs = [
        data_dir / "raw" / "test",
        data_dir / "raw",
        data_dir,
    ]

    with open(txt_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) >= 2:
                img_rel_path = parts[0]
                try:
                    label = int(parts[1])
                    # Strictly filter out Grade 5 (Poor Quality)
                    if label in ignore_classes:
                        continue
                except ValueError:
                    continue

                # Search for the image on disk
                img_file = None
                if (data_dir / img_rel_path).exists():
                    img_file = data_dir / img_rel_path
                else:
                    img_name = Path(img_rel_path).name
                    for search_dir in possible_image_dirs:
                        candidate = search_dir / img_name
                        if candidate.exists():
                            img_file = candidate
                            break

                if img_file and img_file.exists():
                    samples.append((img_file, label))

    return samples


def run_evaluation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"-> Compute Device: {device}")

    # =========================================================================
    # UPDATED PATHS FOR YOUR NEW 512x512 EXPERIMENT
    # =========================================================================
    config_path = PROJECT_ROOT / "configs/experiments/exp_07_convnext_tiny_v4.yaml"

    # Check potential names of the final checkpoint
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

    data_dir = PROJECT_ROOT / "data"
    split_file = "test.txt"

    if checkpoint_path is None or not checkpoint_path.exists():
        print(f"[ERROR] Checkpoint not found in directory: {checkpoint_dir}")
        return

    # 2. Load model and config
    print("[1/3] Loading model architecture & configuration...")
    cfg = ConfigLoader.load(config_path)
    predictor = RetinalPredictor.from_checkpoint(
        checkpoint_path=checkpoint_path, config=cfg, device=device
    )

    # Safely extract architecture information
    arch_name = (
        getattr(cfg.model, "architecture", "convnext_tiny")
        if hasattr(cfg, "model")
        else "Custom Model"
    )
    num_classes = (
        getattr(cfg.model, "num_classes", 5) if hasattr(cfg, "model") else 5
    )

    # Extract image size safely from preprocessing
    img_size = "512x512"
    if hasattr(cfg, "preprocessing"):
        if hasattr(cfg.preprocessing, "resize"):
            img_size = f"{cfg.preprocessing.resize.get('height')}x{cfg.preprocessing.resize.get('width')}"
        elif hasattr(cfg.preprocessing, "img_size"):
            img_size = str(cfg.preprocessing.img_size)

    # 3. Load dataset filtering out Grade 5
    print(f"[2/3] Loading evaluation dataset ({split_file})...")
    samples = load_dataset_split(
        data_dir, split_file=split_file, ignore_classes={5}
    )

    if not samples:
        print(
            f"[ERROR] No valid images loaded from {split_file}. Check your paths."
        )
        return

    print(
        f"      - Total valid test samples (Class 5 filtered out): {len(samples)}"
    )

    # 4. Run Batch Inference
    print("[3/3] Running inference across target test dataset...")
    y_true = []
    y_pred = []

    # Muting internal prints to keep clean progress bar
    with contextlib.redirect_stdout(io.StringIO()):
        for img_path, true_label in tqdm(
            samples, desc="Evaluating", file=sys.__stdout__
        ):
            res = predictor.predict(img_path)
            y_true.append(true_label)
            y_pred.append(res.grade_id)

    # 5. Print Architecture Summary
    print("\n" + "=" * 65)
    print(" MODEL & EXPERIMENT INFORMATION")
    print("=" * 65)
    print(f" Model Architecture : {arch_name}")
    print(f" Config File         : {config_path.name}")
    print(f" Checkpoint Path     : {checkpoint_path}")
    print(f" Target Classes      : {num_classes} (Grades 0, 1, 2, 3, 4)")
    print(f" Input Image Size    : {img_size}")
    print(f" Evaluation Split    : {split_file} (Class 5 Ignored)")
    print("=" * 65)

    # 6. Primary Metric: Quadratic Weighted Kappa (QWK)
    qwk_score = cohen_kappa_score(y_true, y_pred, weights="quadratic")

    # 7. Classification Report (Metrics for Grades 0-4)
    unique_labels = sorted(list(set(y_true)))
    target_names = [f"Grade {lbl}" for lbl in unique_labels]

    print("\n" + "=" * 65)
    print(" CLASSIFICATION METRICS & OVERALL PERFORMANCE")
    print("=" * 65)
    print(f" QUADRATIC WEIGHTED KAPPA (QWK) : {qwk_score:.4f}")
    print("-" * 65)
    report = classification_report(
        y_true,
        y_pred,
        labels=unique_labels,
        target_names=target_names,
        digits=4,
    )
    print(report)

    # 8. Formatted Confusion Matrix
    print("=" * 65)
    print(" CONFUSION MATRIX (Rows: True Grade, Columns: Predicted Grade)")
    print("=" * 65)
    cm = confusion_matrix(y_true, y_pred, labels=unique_labels)

    header = "True \\ Pred\t" + "\t".join([f"G{lbl}" for lbl in unique_labels])
    print(header)
    print("-" * 60)

    for row_label, row in zip(unique_labels, cm, strict=False):
        row_str = "\t".join(map(str, row))
        print(f"Grade {row_label}\t\t{row_str}")

    print("=" * 65)


if __name__ == "__main__":
    run_evaluation()
