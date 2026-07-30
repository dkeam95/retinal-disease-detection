"""CLI Training Entrypoint for Diabetic Retinopathy Lesion Detection.

Usage:
  python scripts/training/train_detection.py --config configs/detection/exp_faster_rcnn_lesions_v3_1536.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import random
import sys

# Ensure src/ is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from common.config.loader import ConfigLoader
from common.config.types import DetectionDatasetConfig, DetectionModelConfig
from dataset.detection_dataset import LesionDetectionDataset, detection_collate_fn
from detection.detector_model import build_lesion_detector
from trainer.detection_trainer import DetectionTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train_detection")


def set_seed(seed: int) -> None:
    """Set global random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Faster R-CNN Lesion Detector")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/detection/exp_faster_rcnn_lesions_v3_1536.yaml",
        help="Path to detection experiment YAML config",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    logger.info(f"Loading Detection Configuration from: {config_path}")
    cfg = ConfigLoader.load(config_path)

    set_seed(cfg.experiment.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() and cfg.training.device == "cuda" else "cpu"
    )
    logger.info(f"Compute Device Target: {device}")

    # Resolve detection dataset config settings
    det_ds_cfg = cfg.detection_dataset or DetectionDatasetConfig(
        xml_dir=Path("data/raw/lesion_detection/train"),
        image_dir=Path("data/raw/train"),
    )
    det_model_cfg = cfg.detection_model or DetectionModelConfig()

    train_xml_dir = cfg.dataset.path / "lesion_detection" / "train"
    train_img_dir = cfg.dataset.path / "train"
    valid_xml_dir = cfg.dataset.path / "lesion_detection" / "valid"
    valid_img_dir = cfg.dataset.path / "valid"

    target_size = (det_model_cfg.min_size, det_model_cfg.max_size)

    logger.info("Initializing Lesion Detection Datasets...")
    train_dataset = LesionDetectionDataset(
        xml_dir=train_xml_dir,
        image_dir=train_img_dir,
        target_size=target_size,
    )
    valid_dataset = LesionDetectionDataset(
        xml_dir=valid_xml_dir,
        image_dir=valid_img_dir,
        target_size=target_size,
    )

    logger.info(f"Train samples: {len(train_dataset)} | Valid samples: {len(valid_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.dataloader.batch_size,
        shuffle=True,
        num_workers=cfg.dataloader.num_workers,
        collate_fn=detection_collate_fn,
        pin_memory=cfg.dataloader.pin_memory,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=cfg.dataloader.batch_size,
        shuffle=False,
        num_workers=cfg.dataloader.num_workers,
        collate_fn=detection_collate_fn,
        pin_memory=cfg.dataloader.pin_memory,
    )

    logger.info(f"Building Faster R-CNN (ResNet50-FPN) detector model [COCO Pretrained: {det_model_cfg.pretrained}]...")
    model = build_lesion_detector(config=det_model_cfg)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.optimizer.lr,
        weight_decay=cfg.optimizer.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.training.epochs,
        eta_min=cfg.scheduler.eta_min,
    )

    trainer = DetectionTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=valid_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        config=cfg,
        device=device,
    )

    result = trainer.train()
    logger.info(f"Detection Training Finished Successfully! Best Checkpoint: {result['best_checkpoint']}")


if __name__ == "__main__":
    main()
