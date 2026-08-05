"""Train Anchor-Free YOLOv8 / YOLOv11 Lesion Detector on 1024x1024 Tiled Dataset."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Anchor-Free YOLO Lesion Detector")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/processed/yolo_lesions_1024/dataset.yaml"),
        help="Path to YOLO dataset.yaml file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8m.pt",
        help="Pretrained YOLO model architecture (yolov8s.pt, yolov8m.pt, yolov8l.pt, etc.)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=40,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for training",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=1024,
        help="Input image tile resolution (default: 1024)",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="experiments/exp_yolov8_lesions_1024",
        help="Project output directory for runs and checkpoints",
    )
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError as err:
        logger.error("ultralytics package is required. Install via `pip install ultralytics`.")
        raise SystemExit(1) from err

    data_path = args.data.resolve()
    if not data_path.exists():
        logger.error(f"Dataset configuration file not found at: {data_path}")
        logger.error("Please run `python scripts/dataset/convert_voc_to_yolo.py` first.")
        raise SystemExit(1)

    logger.info(f"Loading Anchor-Free model: {args.model}")
    model = YOLO(args.model)

    logger.info(f"Starting YOLO training for {args.epochs} epochs on {data_path} at resolution {args.imgsz}x{args.imgsz}...")

    results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        batch=args.batch_size,
        imgsz=args.imgsz,
        project=args.project,
        name="train_run",
        exist_ok=True,
        pretrained=True,
        optimizer="AdamW",
        lr0=0.0001,
        lrf=0.01,
        weight_decay=0.0005,
        mosaic=0.5,     # Subtle spatial augmentation
        mixup=0.0,      # Disabled to avoid blending medical lesion boundaries
        close_mosaic=10,
        verbose=True,
        workers=2,
        device=0 if torch_has_cuda() else "cpu",
    )

    logger.info("YOLO Lesion Detection Training completed successfully!")
    logger.info(f"Best model weights saved to: {Path(args.project) / 'train_run' / 'weights' / 'best.pt'}")


def torch_has_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


if __name__ == "__main__":
    main()
