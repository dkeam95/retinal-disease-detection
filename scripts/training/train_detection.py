"""Train Faster R-CNN lesion detector on retinal dataset."""

from __future__ import annotations

import argparse
import logging
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

from common.config.loader import ConfigLoader
from common.config.types import DetectionModelConfig
from dataset.detection_dataset import LesionDetectionDataset
from detection.detector_model import build_lesion_detector
from trainer.detection_trainer import DetectionTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def collate_fn(batch):
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Faster R-CNN Lesion Detector")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/config.yaml"),
        help="Path to general project config file",
    )
    parser.add_argument(
        "--train-dir",
        type=Path,
        default=Path("data/raw/lesion_detection/train"),
        help="Directory with training XMLs and images",
    )
    parser.add_argument(
        "--val-dir",
        type=Path,
        default=Path("data/raw/lesion_detection/valid"),
        help="Directory with validation XMLs and images",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of epochs to train for",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size for training",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--min-size",
        type=int,
        default=640,
        help="Faster R-CNN input min size resolution",
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=853,
        help="Faster R-CNN input max size resolution",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to train on",
    )
    args = parser.parse_args()

    device = torch.device(args.device)

    # 1. Load config
    from dataclasses import replace
    config = ConfigLoader.load(args.config)
    config = replace(
        config,
        training=replace(config.training, epochs=args.epochs),
        dataloader=replace(config.dataloader, batch_size=args.batch_size),
        detection_model=DetectionModelConfig(
            architecture="fasterrcnn_resnet50_fpn",
            pretrained=True,
            num_classes=5,
            score_thresh=0.25,
            nms_thresh=0.45,
            min_size=args.min_size,
            max_size=args.max_size,
        ),
    )


    # 2. Build Datasets
    logger.info(f"Loading training data from: {args.train_dir}")
    train_dataset = LesionDetectionDataset(
        xml_dir=args.train_dir,
        image_dir=args.train_dir,
        enable_clahe=True,
    )

    logger.info(f"Loading validation data from: {args.val_dir}")
    val_dataset = LesionDetectionDataset(
        xml_dir=args.val_dir,
        image_dir=args.val_dir,
        enable_clahe=True,
    )

    # 3. Build DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    # 4. Instantiate Model
    logger.info(f"Building Faster R-CNN with min_size={args.min_size}, max_size={args.max_size}")
    model = build_lesion_detector(config=config.detection_model)

    # 5. Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 6. Run Trainer
    trainer = DetectionTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        device=device,
    )

    results = trainer.train()
    logger.info(f"Training completed successfully! Best mAP@50: {results['best_map_50']:.4f}")
    logger.info(f"Best model saved to: {results['best_checkpoint']}")


if __name__ == "__main__":
    main()
