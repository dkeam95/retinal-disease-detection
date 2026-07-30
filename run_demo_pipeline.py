"""Live demonstration of the complete Retinal Disease Detection Pipeline.

Steps:
    1. Multi-Model Ensemble Prediction (ConvNeXt-Tiny v2 + Swin-T + DenseNet-121 v2)
    2. Ensemble inference on a single sample image
    3. Error Analysis + Confusion Matrix on a test subset
    4. CLAHE + Spotlight Grad-CAM Explainability Gallery
"""

from __future__ import annotations

import os

# Suppress albumentations update check warnings (fires in every DataLoader worker on Windows)
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

from pathlib import Path

import numpy as np
import torch

from common.config.loader import ConfigLoader
from common.config.types import DataLoaderConfig, DatasetConfig
from dataloader.factory import build_validation_dataloader
from dataset.dataset import RetinalDataset
from evaluation.error_analysis import ErrorAnalyzer
from explainability.batch_audit import GradCAMBatchAuditor
from inference.ensemble import EnsemblePredictor
from inference.predictor import RetinalPredictor


# ── Configuration ──────────────────────────────────────────────────────────────
CONFIG_DIR = Path("configs/experiments")
EXPERIMENTS_DIR = Path("experiments")

MODELS = [
    (
        "exp_05_convnext_tiny_v2",
        EXPERIMENTS_DIR / "exp_05_convnext_tiny_v2" / "checkpoints" / "best.pt",
        0.40,
    ),
    (
        "exp_07_swin_t_v2",
        EXPERIMENTS_DIR / "exp_07_swin_t_v2" / "checkpoints" / "best.pt",
        0.35,
    ),
    (
        "exp_02_densenet121_v2",
        EXPERIMENTS_DIR / "exp_02_densenet121_v2" / "checkpoints" / "best.pt",
        0.25,
    ),
]

SAMPLE_IMAGE = Path("data/raw/test/007-2809-100.jpg")
TEST_ANNO_FILE = "test.txt"
TEST_IMG_DIR = "test"
DATA_ROOT = Path("data/raw")

DEMO_BATCH_LIMIT = 6  # ~6 x 32 = 192 images for evaluation
MAX_SAMPLES_PER_CLASS = 2  # Grad-CAM: up to 2 images per DR grade
GRADCAM_MODE = "4panel"  # "heatmap" | "spotlight" | "4panel"


# ── Helpers ────────────────────────────────────────────────────────────────────
def _sep(title: str) -> None:
    print(f"\n{'=' * 70}\n  {title}\n{'=' * 70}")


# ── Step functions ─────────────────────────────────────────────────────────────


def load_ensemble(
    device: torch.device,
) -> tuple[list[RetinalPredictor], EnsemblePredictor]:
    """Load checkpoints and build weighted ensemble."""
    predictors: list[RetinalPredictor] = []
    weights: list[float] = []

    for exp_name, ckpt_path, weight in MODELS:
        cfg_path = CONFIG_DIR / f"{exp_name}.yaml"
        if not cfg_path.exists():
            print(f"  [SKIP] Config not found : {cfg_path}")
            continue
        if not ckpt_path.exists():
            print(f"  [SKIP] Checkpoint not found : {ckpt_path}")
            continue

        config = ConfigLoader.load(cfg_path)
        predictor = RetinalPredictor.from_checkpoint(
            checkpoint_path=ckpt_path,
            config=config,
            device=device,
        )
        predictors.append(predictor)
        weights.append(weight)
        print(f"  [OK] {config.model.architecture:<20} <- {ckpt_path}")

    if not predictors:
        raise RuntimeError("No valid checkpoints found. Check MODELS paths above.")

    ensemble = EnsemblePredictor(predictors=predictors, weights=weights)
    return predictors, ensemble


def run_single_inference(ensemble: EnsemblePredictor) -> None:
    """Ensemble prediction on one sample image."""
    if not SAMPLE_IMAGE.exists():
        print(f"  [SKIP] Sample image not found: {SAMPLE_IMAGE}")
        return

    result = ensemble.predict(SAMPLE_IMAGE)
    print(f"  Image           : {SAMPLE_IMAGE.name}")
    print(f"  Predicted Grade : Grade {result.grade_id} - {result.grade_name}")
    print(f"  Confidence      : {result.confidence:.2%}")
    print("  Class Distribution:")
    for i, (name, prob) in enumerate(zip(ensemble.class_names, result.probabilities)):
        bar = "#" * int(prob * 30)
        print(f"    Class {i} ({name:<18}): {prob:6.2%}  {bar}")


def run_error_analysis(
    predictors: list[RetinalPredictor],
    device: torch.device,
) -> list:
    """Evaluate on test subset; return sample_batches for Grad-CAM reuse."""
    anno_path = DATA_ROOT / TEST_ANNO_FILE
    img_dir = DATA_ROOT / TEST_IMG_DIR

    if not anno_path.exists() or not img_dir.exists():
        print(f"  [SKIP] Test split not found: {anno_path}")
        return []

    test_cfg = DatasetConfig(
        path=DATA_ROOT,
        annotation_file=TEST_ANNO_FILE,
        image_directory=TEST_IMG_DIR,
        num_classes=5,
    )
    test_dataset = RetinalDataset(config=test_cfg)

    # num_workers=0 is critical on Windows — avoids subprocess spawn overhead
    loader_cfg = DataLoaderConfig(
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
        persistent_workers=False,
    )
    test_loader = build_validation_dataloader(dataset=test_dataset, config=loader_cfg)

    print(f"  Dataset size   : {len(test_dataset)} images")
    print(
        f"  Evaluating     : first {DEMO_BATCH_LIMIT} batches (~{DEMO_BATCH_LIMIT * 32} images)"
    )

    y_true: list[int] = []
    y_pred: list[int] = []
    sample_batches: list = []

    for idx, batch in enumerate(test_loader):
        if idx >= DEMO_BATCH_LIMIT:
            break
        sample_batches.append(batch)

        imgs = batch.image.to(device)
        targets = batch.label.numpy()

        batch_probs = []
        for p in predictors:
            with torch.no_grad():
                outputs = p.model(imgs)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs
                probs = torch.softmax(logits, dim=1).cpu().numpy()
            batch_probs.append(probs)

        ensemble_probs = np.mean(batch_probs, axis=0)
        preds = np.argmax(ensemble_probs, axis=1)
        y_true.extend(targets.tolist())
        y_pred.extend(preds.tolist())

    analyzer = ErrorAnalyzer()
    analyzer.generate_full_error_report(
        y_true=y_true,
        y_pred=y_pred,
        output_dir="reports/error_analysis",
        experiment_name="live_ensemble_demo",
    )
    print(
        "  [OK] Confusion matrix  -> reports/error_analysis/live_ensemble_demo/confusion_matrix.png"
    )
    print(
        "  [OK] JSON error report -> reports/error_analysis/live_ensemble_demo/error_report.json"
    )
    return sample_batches


def run_gradcam_audit(
    predictors: list[RetinalPredictor],
    sample_batches: list,
    mode: str = "4panel",
) -> None:
    """Generate CLAHE + Spotlight Grad-CAM gallery from pre-collected sample batches."""
    if not sample_batches:
        print("  [SKIP] No sample batches available for Grad-CAM.")
        return

    auditor = GradCAMBatchAuditor(
        predictor=predictors[0],
        output_dir="reports/figures/gradcam_audit",
        spotlight_threshold=0.45,
        panel_size=(320, 320),
    )
    result = auditor.run_audit(
        dataloader=sample_batches,
        experiment_name="live_demo",
        max_samples_per_class=MAX_SAMPLES_PER_CLASS,
        mode=mode,
    )
    print(f"  [OK] Mode          : {mode}")
    print(f"  [OK] Samples saved : {result['total_samples']} images")
    print(f"  [OK] HTML gallery  : {result['gallery_html']}")


# ── Main ───────────────────────────────────────────────────────────────────────


def main() -> None:
    _sep("RETINAL DISEASE DETECTION - LIVE PIPELINE DEMO")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device : {device}")

    _sep("Step 1/4 - Loading Model Ensemble")
    predictors, ensemble = load_ensemble(device)
    print(f"  Ensemble ready : {len(predictors)} models loaded")

    _sep("Step 2/4 - Single Image Ensemble Inference")
    run_single_inference(ensemble)

    _sep("Step 3/4 - Error Analysis & Confusion Matrix")
    sample_batches = run_error_analysis(predictors, device)

    _sep("Step 4/4 - CLAHE + Spotlight Grad-CAM Gallery")
    run_gradcam_audit(predictors, sample_batches, mode=GRADCAM_MODE)

    _sep("DEMO COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()
