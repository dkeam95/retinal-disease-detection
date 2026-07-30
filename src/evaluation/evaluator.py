"""Independent Model Evaluator module."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader

from common.config.types import ProjectConfig
from training.model_factory import ModelFactory
from visualization.report_generator import HTMLReportGenerator

DEFAULT_CLASS_NAMES = [
    "No DR",
    "Mild NPDR",
    "Moderate NPDR",
    "Severe NPDR",
    "Proliferative DR",
]


class ModelEvaluator:
    """Evaluates fine-tuned neural network models on validation/test datasets independently."""

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device | str = "cpu",
        class_names: list[str] | None = None,
    ) -> None:
        """Initialize Evaluator.

        Args:
            model: PyTorch model instance.
            device: Computing device.
            class_names: List of target class names.
        """
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.model.eval()
        self.class_names = (
            class_names if class_names is not None else DEFAULT_CLASS_NAMES
        )

    @classmethod
    def from_checkpoint(
        self,
        checkpoint_path: str | Path,
        config: ProjectConfig,
        device: torch.device | str = "cpu",
    ) -> ModelEvaluator:
        """Instantiate evaluator directly from a saved checkpoint file path and project config.

        Args:
            checkpoint_path: Path to PyTorch model checkpoint (.pt).
            config: Project configuration object.
            device: Execution compute device.

        Returns:
            Configured ModelEvaluator instance.
        """
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {path}")

        dev = torch.device(device)
        model = ModelFactory.build(
            architecture=config.model.architecture,
            pretrained=False,
            num_classes=config.model.num_classes,
            dropout_rate=config.model.dropout_rate,
        )

        checkpoint_data = torch.load(path, map_location=dev, weights_only=False)
        state_dict = (
            checkpoint_data["model_state"]
            if isinstance(checkpoint_data, dict) and "model_state" in checkpoint_data
            else checkpoint_data
        )
        model.load_state_dict(state_dict)

        return ModelEvaluator(model=model, device=dev)

    @torch.no_grad()
    def evaluate(
        self,
        dataloader: DataLoader,
        save_report_path: str | Path | None = None,
        save_json_path: str | Path | None = None,
        experiment_name: str = "Evaluation Run",
    ) -> dict[str, Any]:
        """Execute evaluation loop over dataloader and calculate metrics.

        Args:
            dataloader: Test or validation DataLoader.
            save_report_path: Optional path to save HTML report.
            save_json_path: Optional path to save raw JSON metrics.
            experiment_name: Identifier string for reporting.

        Returns:
            Dictionary containing qwk, accuracy, macro_f1, per_class, and confusion_matrix.
        """
        all_preds = []
        all_targets = []

        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                images, targets = batch[0], batch[1]
            elif isinstance(batch, dict):
                images, targets = batch["image"], batch["label"]
            else:
                images, targets = batch.images, batch.labels

            images = images.to(self.device)
            outputs = self.model(images)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs

            preds = torch.argmax(logits, dim=1).cpu().numpy()
            targets_np = (
                targets.cpu().numpy()
                if isinstance(targets, torch.Tensor)
                else np.array(targets)
            )

            all_preds.extend(preds)
            all_targets.extend(targets_np)

        preds_arr = np.array(all_preds)
        targets_arr = np.array(all_targets)

        # Calculate metrics using sklearn.metrics
        qwk = float(cohen_kappa_score(targets_arr, preds_arr, weights="quadratic"))
        acc = float(accuracy_score(targets_arr, preds_arr))
        macro_f1 = float(
            f1_score(targets_arr, preds_arr, average="macro", zero_division=0)
        )
        weighted_f1 = float(
            f1_score(targets_arr, preds_arr, average="weighted", zero_division=0)
        )
        cm = confusion_matrix(
            targets_arr, preds_arr, labels=list(range(len(self.class_names)))
        )

        # Calculate per-class metrics
        precisions = precision_score(
            targets_arr, preds_arr, average=None, zero_division=0
        )
        recalls = recall_score(targets_arr, preds_arr, average=None, zero_division=0)
        f1s = f1_score(targets_arr, preds_arr, average=None, zero_division=0)

        per_class_dict = {}
        for i, name in enumerate(self.class_names):
            per_class_dict[name] = {
                "precision": float(precisions[i]) if i < len(precisions) else 0.0,
                "recall": float(recalls[i]) if i < len(recalls) else 0.0,
                "f1": float(f1s[i]) if i < len(f1s) else 0.0,
            }

        results = {
            "qwk": qwk,
            "accuracy": acc,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "confusion_matrix": cm.tolist(),
            "per_class": per_class_dict,
        }

        # Export JSON report
        if save_json_path is not None:
            json_file = Path(save_json_path)
            json_file.parent.mkdir(parents=True, exist_ok=True)
            with json_file.open("w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)

        # Export HTML report
        if save_report_path is not None:
            HTMLReportGenerator.generate(
                experiment_name=experiment_name,
                metrics=results,
                confusion_matrix=cm,
                per_class_metrics=per_class_dict,
                output_path=save_report_path,
            )

        return results
