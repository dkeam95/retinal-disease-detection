"""Detection Trainer for Faster R-CNN Lesion Detection."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from common.config.types import ProjectConfig
from detection.metrics import LesionDetectionEvaluator

logger = logging.getLogger(__name__)


class DetectionTrainer:
    """Trainer managing object detection training, validation, AMP mixed precision, and checkpointing."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader[Any],
        val_loader: DataLoader[Any],
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        config: ProjectConfig,
        device: torch.device,
    ) -> None:
        """Initialize DetectionTrainer.

        Args:
            model: PyTorch Faster R-CNN model.
            train_loader: DataLoader for training set.
            val_loader: DataLoader for validation set.
            optimizer: PyTorch optimizer.
            scheduler: Learning rate scheduler.
            config: Aggregated ProjectConfig dataclass.
            device: Computing device (CUDA / CPU).
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = device

        self.use_amp = config.mixed_precision.enabled and device.type == "cuda"
        self.scaler = GradScaler(enabled=self.use_amp)
        self.evaluator = LesionDetectionEvaluator()

        self.checkpoint_dir = Path(config.checkpoint.directory)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_map_50 = 0.0

    def train_epoch(self, epoch: int) -> dict[str, float]:
        """Run one training epoch.

        Args:
            epoch: Current epoch index.

        Returns:
            Dictionary containing average training losses.
        """
        self.model.train()
        total_loss = 0.0
        loss_dict_sum: dict[str, float] = {}

        num_batches = len(self.train_loader)

        for step, (images, targets) in enumerate(self.train_loader, start=1):
            images = [img.to(self.device) for img in images]
            targets = [
                {k: v.to(self.device) for k, v in t.items()} for t in targets
            ]

            self.optimizer.zero_grad()

            with autocast(enabled=self.use_amp):
                # Faster R-CNN in train mode returns a dict of losses
                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

            self.scaler.scale(losses).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            batch_loss = float(losses.item())
            total_loss += batch_loss

            for k, v in loss_dict.items():
                loss_dict_sum[k] = loss_dict_sum.get(k, 0.0) + float(v.item())

            if step % self.config.logging.log_every_n_steps == 0 or step == num_batches:
                logger.info(
                    f"Epoch [{epoch}/{self.config.training.epochs}] Step [{step}/{num_batches}] - "
                    f"Loss: {batch_loss:.4f}"
                )

        avg_loss = total_loss / max(num_batches, 1)
        avg_losses = {
            "loss": avg_loss,
            **{k: v / max(num_batches, 1) for k, v in loss_dict_sum.items()},
        }
        return avg_losses

    @torch.no_grad()
    def validate(self) -> dict[str, float]:
        """Run validation step and compute mAP metrics.

        Returns:
            Dictionary of computed mAP metrics.
        """
        self.model.eval()
        self.evaluator.reset()

        for images, targets in self.val_loader:
            images = [img.to(self.device) for img in images]
            targets = [
                {k: v.to(self.device) for k, v in t.items()} for t in targets
            ]

            # Faster R-CNN in eval mode returns predictions: list of dicts with 'boxes', 'scores', 'labels'
            predictions = self.model(images)
            self.evaluator.update(predictions, targets)

        metrics = self.evaluator.compute()
        return metrics

    def train(self) -> dict[str, Any]:
        """Run full training pipeline across all epochs.

        Returns:
            Dictionary summarizing overall training metrics and checkpoint path.
        """
        logger.info(f"Starting Detection Training for {self.config.training.epochs} epochs...")

        history: list[dict[str, Any]] = []

        for epoch in range(1, self.config.training.epochs + 1):
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate()

            if self.scheduler is not None:
                self.scheduler.step()

            map_50 = val_metrics.get("map_50", 0.0)
            logger.info(
                f"Epoch [{epoch}/{self.config.training.epochs}] Completed - "
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Val mAP@50: {map_50:.4f} | Val mAP: {val_metrics.get('map', 0.0):.4f}"
            )

            epoch_record = {"epoch": epoch, **train_metrics, **val_metrics}
            history.append(epoch_record)

            # Checkpoint best model based on mAP_50
            if map_50 >= self.best_map_50:
                self.best_map_50 = map_50
                best_ckpt_path = self.checkpoint_dir / "best.pt"
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "best_map_50": self.best_map_50,
                        "config": self.config,
                    },
                    best_ckpt_path,
                )
                logger.info(f"-> Saved new best checkpoint [mAP@50: {map_50:.4f}] to {best_ckpt_path}")

        # Save final checkpoint
        latest_ckpt_path = self.checkpoint_dir / "latest.pt"
        torch.save(
            {
                "epoch": self.config.training.epochs,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_map_50": self.best_map_50,
                "config": self.config,
            },
            latest_ckpt_path,
        )

        return {
            "best_map_50": self.best_map_50,
            "best_checkpoint": str(self.checkpoint_dir / "best.pt"),
            "history": history,
        }
