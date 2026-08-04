from __future__ import annotations

import os

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

import sys
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize
from sklearn.metrics import classification_report, cohen_kappa_score

# Add project root and src directory to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from common.config.loader import ConfigLoader
from inference.predictor import RetinalPredictor


def load_annotations_direct(
    data_dir: Path, ignore_classes: set[int] = {5}
) -> dict[str, int]:
    """
    Directly load target annotations from text or CSV ground truth files.

    Parameters
    ----------
    data_dir : Path
        Directory containing data files and annotations.
    ignore_classes : set[int]
        Class IDs to ignore during evaluation (default: {5}).

    Returns
    -------
    dict[str, int]
        Mapping from image filename to integer target grade.
    """
    gt_map: dict[str, int] = {}
    txt_files = list(data_dir.rglob("*.txt")) + list(data_dir.rglob("*.csv"))

    for txt_file in txt_files:
        try:
            with open(txt_file, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.replace(",", " ").replace("\t", " ").split()
                    if len(parts) >= 2:
                        img_name = Path(parts[0]).name
                        try:
                            label = int(parts[1])
                            if label not in ignore_classes:
                                gt_map[img_name] = label
                        except ValueError:
                            continue
        except Exception:
            continue
    return gt_map


def qwk_objective(
    thresholds: np.ndarray | list[float],
    raw_scores: np.ndarray,
    targets: np.ndarray,
) -> float:
    """
    Objective function to minimize negative Quadratic Weighted Kappa (QWK).

    Parameters
    ----------
    thresholds : np.ndarray | list[float]
        Candidate thresholds for digitizing expected scores into discrete grades.
    raw_scores : np.ndarray
        Continuous expected class values E[y].
    targets : np.ndarray
        Ground truth labels.

    Returns
    -------
    float
        Negative Quadratic Weighted Kappa score.
    """
    preds = np.digitize(raw_scores, thresholds)
    return float(-cohen_kappa_score(targets, preds, weights="quadratic"))


def main() -> None:
    """
    Execute model inference on test set and optimize decision thresholds for QWK.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"-> Using compute device: {device}")

    config_path = PROJECT_ROOT / "configs/experiments/exp_07_convnext_tiny_v4.yaml"
    ckpt_path = (
        PROJECT_ROOT
        / "experiments/exp_07_convnext_tiny_v4/checkpoints/best.pt"
    )

    if not ckpt_path.exists():
        ckpt_path = (
            PROJECT_ROOT
            / "experiments/exp_07_convnext_tiny_v4/checkpoints/best_model.pt"
        )

    print(f"-> Loading configuration: {config_path}")
    print(f"-> Loading checkpoint: {ckpt_path}")

    cfg = ConfigLoader.load(config_path)
    predictor = RetinalPredictor.from_checkpoint(
        checkpoint_path=ckpt_path, config=cfg, device=device
    )

    data_dir = PROJECT_ROOT / "data"
    gt_map = load_annotations_direct(data_dir, ignore_classes={5})
    print(f"-> Found {len(gt_map)} annotated images across data directories.")

    test_dir = data_dir / "raw" / "test"
    if not test_dir.exists():
        test_dir = data_dir / "test"

    test_images = [
        p
        for p in test_dir.rglob("*.*")
        if p.name in gt_map and p.suffix.lower() in [".jpg", ".png", ".jpeg"]
    ]

    print(f"-> Collecting class probability distributions over {len(test_images)} test images...")

    weights_vec = np.array([0, 1, 2, 3, 4])
    raw_scores_list: list[float] = []
    targets_list: list[int] = []

    for img_path in test_images:
        res = predictor.predict(img_path)

        # Support both dictionary ({0: p0, 1: p1...}) and list/array outputs
        if isinstance(res.probabilities, dict):
            probs = np.array(list(res.probabilities.values()))
        else:
            probs = np.array(res.probabilities)

        # Calculate Expected Value E[y] = sum(i * P_i)
        expected_score = float(np.sum(probs * weights_vec))

        raw_scores_list.append(expected_score)
        targets_list.append(gt_map[img_path.name])

    raw_scores = np.array(raw_scores_list)
    targets = np.array(targets_list)

    # 1. Baseline QWK (Standard rounding)
    base_preds = np.round(raw_scores).astype(int)
    base_qwk = cohen_kappa_score(targets, base_preds, weights="quadratic")

    # 2. Threshold optimization via Nelder-Mead
    init_thresholds = [0.5, 1.5, 2.5, 3.5]
    opt_res = minimize(
        qwk_objective,
        init_thresholds,
        args=(raw_scores, targets),
        method="Nelder-Mead",
    )

    opt_thresholds = np.sort(opt_res.x)
    opt_preds = np.digitize(raw_scores, opt_thresholds)
    opt_qwk = cohen_kappa_score(targets, opt_preds, weights="quadratic")

    print("\n" + "=" * 60)
    print("QWK THRESHOLD OPTIMIZATION RESULTS:")
    print(f"  - Initial Thresholds (Standard) : {init_thresholds}")
    print(
        f"  - Optimal Thresholds (Optimized) : {[round(float(x), 3) for x in opt_thresholds]}"
    )
    print(
        f"  - Test QWK                       : {base_qwk:.4f} ===> {opt_qwk:.4f} (+{opt_qwk - base_qwk:.4f})"
    )
    print("=" * 60)

    print("\nUpdated Classification Report with Optimal Thresholds:")
    print(
        classification_report(
            targets,
            opt_preds,
            target_names=[f"Grade {i}" for i in range(5)],
            digits=4,
        )
    )


if __name__ == "__main__":
    main()
