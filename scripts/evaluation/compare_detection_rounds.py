"""Compare three rounds of lesion detection models and print mAP leaderboard."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from checkpoint.utils import clean_state_dict
from dataset.detection_dataset import LesionDetectionDataset
from detection.detector_model import build_lesion_detector
from detection.metrics import LesionDetectionEvaluator


def collate_fn(batch):
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Faster R-CNN Detection Rounds")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw/lesion_detection/test"),
        help="Directory containing test XMLs and images",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Compute device",
    )
    args = parser.parse_known_args()[0]

    device = torch.device(args.device)

    # 1. Resolve checkpoints for the 3 rounds
    checkpoints = {
        "Round 1 (640)": Path("experiments/exp_faster_rcnn_lesions/checkpoints/best.pt"),
        "Round 2 (1024)": Path("experiments/exp_faster_rcnn_lesions_v2_1024_legacy/checkpoints/best.pt"),
        "Round 3 (1536)": Path("experiments/exp_faster_rcnn_lesions_v3_1536/checkpoints/best.pt"),
    }

    # 2. Build Dataset & DataLoader
    dataset = LesionDetectionDataset(
        xml_dir=args.data_dir,
        image_dir=args.data_dir,
        enable_clahe=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    print(f"[DETECTION-COMPARE] Loaded test dataset with {len(dataset)} samples.")
    print("-" * 70)
    print(f"{'Model Round':<18} | {'mAP@50':<10} | {'mAP@75':<10} | {'mAP@50:95':<10}")
    print("-" * 70)

    evaluator = LesionDetectionEvaluator()

    for round_name, ckpt_path in checkpoints.items():
        if not ckpt_path.exists():
            print(f"{round_name:<18} | [NOT FOUND at {ckpt_path}]")
            continue

        # Load checkpoint
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

        # Detect resolution configurations from checkpoint or fallback defaults
        min_size = 1536
        max_size = 2048
        if "1024" in str(ckpt_path):
            min_size = 1024
            max_size = 1365
        elif "exp_faster_rcnn_lesions/" in str(ckpt_path):
            min_size = 640
            max_size = 853

        # Instantiate Faster R-CNN
        model = build_lesion_detector(
            min_size=min_size,
            max_size=max_size,
            pretrained=False,
        )
        model.load_state_dict(clean_state_dict(ckpt["model_state_dict"]))
        model.to(device)
        model.eval()

        evaluator.reset()

        with torch.no_grad():
            for images, targets in loader:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                predictions = model(images)
                evaluator.update(predictions, targets)

        metrics = evaluator.compute()
        map_50 = metrics.get("map_50", 0.0)
        map_75 = metrics.get("map_75", 0.0)
        map_50_95 = metrics.get("map", 0.0)

        print(f"{round_name:<18} | {map_50:.4f}     | {map_75:.4f}     | {map_50_95:.4f}")

    print("-" * 70)


if __name__ == "__main__":
    main()
