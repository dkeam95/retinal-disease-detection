"""Detailed Error Analysis & Confusion Matrix Reporting for Retinal Disease Classification."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from evaluation.evaluator import DEFAULT_CLASS_NAMES


class ErrorAnalyzer:
    """Analyzes prediction outputs against ground truth labels and exports visualizations."""

    def __init__(self, class_names: list[str] | None = None) -> None:
        """Initialize ErrorAnalyzer.

        Args:
            class_names: List of class description labels.
        """
        self.class_names = (
            class_names if class_names is not None else DEFAULT_CLASS_NAMES
        )

    def compute_confusion_matrix(
        self,
        y_true: np.ndarray | list[int],
        y_pred: np.ndarray | list[int],
        normalize: str | None = None,
    ) -> np.ndarray:
        """Compute 5x5 confusion matrix.

        Args:
            y_true: Ground truth target array.
            y_pred: Predicted class array.
            normalize: Normalization mode ('true', 'pred', 'all', or None).

        Returns:
            2D numpy array confusion matrix.
        """
        return confusion_matrix(
            y_true,
            y_pred,
            labels=list(range(len(self.class_names))),
            normalize=normalize,
        )

    def plot_confusion_matrix(
        self,
        y_true: np.ndarray | list[int],
        y_pred: np.ndarray | list[int],
        save_path: str | Path | None = None,
        title: str = "Retinal Disease Detection - Confusion Matrix",
    ) -> plt.Figure:
        """Plot dual raw-count and percentage-normalized confusion matrix heatmaps.

        Args:
            y_true: Ground truth target array.
            y_pred: Predicted class array.
            save_path: Optional path to save PNG figure.
            title: Chart title.

        Returns:
            Matplotlib Figure instance.
        """
        cm_counts = self.compute_confusion_matrix(y_true, y_pred, normalize=None)
        cm_norm = self.compute_confusion_matrix(y_true, y_pred, normalize="true")

        fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))

        # Raw Counts Heatmap
        im0 = axes[0].imshow(cm_counts, cmap="Blues")
        axes[0].set_title("Absolute Counts", fontsize=12, fontweight="bold")
        axes[0].set_xlabel("Predicted Label", fontsize=10, fontweight="bold")
        axes[0].set_ylabel("True Ground Truth Label", fontsize=10, fontweight="bold")
        axes[0].set_xticks(range(len(self.class_names)))
        axes[0].set_yticks(range(len(self.class_names)))
        axes[0].set_xticklabels(self.class_names, rotation=20, ha="right")
        axes[0].set_yticklabels(self.class_names)
        for i in range(len(self.class_names)):
            for j in range(len(self.class_names)):
                axes[0].text(
                    j,
                    i,
                    str(cm_counts[i, j]),
                    ha="center",
                    va="center",
                    color="white" if cm_counts[i, j] > cm_counts.max() / 2 else "black",
                    fontweight="bold",
                )

        # Normalized Percentage Heatmap
        im1 = axes[1].imshow(cm_norm, cmap="Greens", vmin=0.0, vmax=1.0)
        axes[1].set_title("Normalized Recall (Row %)", fontsize=12, fontweight="bold")
        axes[1].set_xlabel("Predicted Label", fontsize=10, fontweight="bold")
        axes[1].set_ylabel("True Ground Truth Label", fontsize=10, fontweight="bold")
        axes[1].set_xticks(range(len(self.class_names)))
        axes[1].set_yticks(range(len(self.class_names)))
        axes[1].set_xticklabels(self.class_names, rotation=20, ha="right")
        axes[1].set_yticklabels(self.class_names)
        for i in range(len(self.class_names)):
            for j in range(len(self.class_names)):
                axes[1].text(
                    j,
                    i,
                    f"{cm_norm[i, j]:.1%}",
                    ha="center",
                    va="center",
                    color="white" if cm_norm[i, j] > 0.5 else "black",
                    fontweight="bold",
                )

        fig.suptitle(title, fontsize=15, fontweight="bold", y=1.02)
        plt.tight_layout()

        if save_path is not None:
            path = Path(save_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=300, bbox_inches="tight")

        return fig

    def generate_full_error_report(
        self,
        y_true: np.ndarray | list[int],
        y_pred: np.ndarray | list[int],
        output_dir: str | Path = "reports/error_analysis",
        experiment_name: str = "ensemble_best",
    ) -> dict[str, Any]:
        """Generate comprehensive error report including JSON summary and confusion matrix PNG.

        Args:
            y_true: Ground truth target array.
            y_pred: Predicted class array.
            output_dir: Target output folder.
            experiment_name: Run identifier.

        Returns:
            Dictionary summary of error report.
        """
        out_dir = Path(output_dir) / experiment_name
        out_dir.mkdir(parents=True, exist_ok=True)

        fig_path = out_dir / "confusion_matrix.png"
        self.plot_confusion_matrix(
            y_true,
            y_pred,
            save_path=fig_path,
            title=f"Confusion Matrix - {experiment_name}",
        )

        labels = list(range(len(self.class_names)))
        report_dict = classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=self.class_names,
            output_dict=True,
            zero_division=0,
        )

        cm = self.compute_confusion_matrix(y_true, y_pred, normalize=None)
        json_summary = {
            "experiment_name": experiment_name,
            "confusion_matrix": cm.tolist(),
            "classification_report": report_dict,
        }

        json_path = out_dir / "error_report.json"
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(json_summary, f, indent=2)

        return json_summary
