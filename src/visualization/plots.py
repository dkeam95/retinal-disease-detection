"""Visualization utilities for training metrics, confusion matrix, and class performance."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

DEFAULT_CLASS_NAMES = [
    "No DR",
    "Mild NPDR",
    "Moderate NPDR",
    "Severe NPDR",
    "Proliferative DR",
]


def plot_learning_curves(
    history: dict[str, list[float]],
    save_path: str | Path | None = None,
    title: str = "Training & Validation Performance Curves",
) -> plt.Figure:
    """Plot Loss, QWK, Accuracy, and auxiliary metrics over epochs.

    Args:
        history: Dictionary containing epoch metrics history (e.g., 'train_loss', 'val_loss', 'val_qwk', 'val_accuracy').
        save_path: Optional path to save PNG figure.
        title: Plot super title.

    Returns:
        Matplotlib Figure object.
    """
    epochs = range(1, len(history.get("train_loss", history.get("val_loss", []))) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # 1. Loss curves
    ax_loss = axes[0]
    if "train_loss" in history:
        ax_loss.plot(epochs, history["train_loss"], label="Train Loss", color="#1f77b4", linewidth=2, marker="o")
    if "val_loss" in history:
        ax_loss.plot(epochs, history["val_loss"], label="Val Loss", color="#ff7f0e", linewidth=2, marker="s")
    ax_loss.set_title("Loss Trajectory", fontsize=12)
    ax_loss.set_xlabel("Epoch", fontsize=10)
    ax_loss.set_ylabel("Loss", fontsize=10)
    ax_loss.grid(True, linestyle="--", alpha=0.6)
    ax_loss.legend(fontsize=10)

    # 2. Metrics curves (QWK / Accuracy / F1)
    ax_metric = axes[1]
    color_map = {"val_qwk": "#2ca02c", "val_accuracy": "#d62728", "val_macro_f1": "#9467bd"}
    for metric_name in ["val_qwk", "val_accuracy", "val_macro_f1"]:
        if metric_name in history:
            label = metric_name.replace("val_", "").upper()
            ax_metric.plot(
                epochs,
                history[metric_name],
                label=label,
                color=color_map.get(metric_name, "#17becf"),
                linewidth=2,
                marker="^",
            )

    ax_metric.set_title("Validation Evaluation Metrics", fontsize=12)
    ax_metric.set_xlabel("Epoch", fontsize=10)
    ax_metric.set_ylabel("Score", fontsize=10)
    ax_metric.set_ylim(0.0, 1.05)
    ax_metric.grid(True, linestyle="--", alpha=0.6)
    ax_metric.legend(fontsize=10)

    plt.tight_layout()

    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300, bbox_inches="tight")

    return fig


def plot_confusion_matrix(
    cm: np.ndarray | list[list[int]],
    class_names: list[str] | None = None,
    save_path: str | Path | None = None,
    normalize: bool = True,
    title: str = "Confusion Matrix",
) -> plt.Figure:
    """Plot 5x5 heatmap of confusion matrix for DR severity grading using Matplotlib.

    Args:
        cm: Square confusion matrix array.
        class_names: Class labels. Defaults to 5 DR severity stages.
        save_path: Optional path to save figure.
        normalize: Whether to normalize matrix values by row sums.
        title: Plot title.

    Returns:
        Matplotlib Figure object.
    """
    matrix = np.array(cm, dtype=np.float64)
    names = class_names if class_names is not None else DEFAULT_CLASS_NAMES

    if normalize:
        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        matrix_display = matrix / row_sums
    else:
        matrix_display = matrix

    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(matrix_display, interpolation="nearest", cmap=plt.cm.Blues)
    fig.colorbar(im, ax=ax)

    tick_marks = np.arange(len(names))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=10)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(names, fontsize=10)

    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Grade", fontsize=11, labelpad=8)
    ax.set_ylabel("True Grade", fontsize=11, labelpad=8)

    # Annotate matrix values
    thresh = matrix_display.max() / 2.0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix_display[i, j]
            text = f"{val:.1%}" if normalize else f"{int(matrix[i, j])}"
            color = "white" if val > thresh else "black"
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=10, fontweight="bold")

    plt.tight_layout()

    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300, bbox_inches="tight")

    return fig


def plot_per_class_metrics(
    per_class_data: dict[str, dict[str, float]] | dict[str, Any],
    class_names: list[str] | None = None,
    save_path: str | Path | None = None,
    title: str = "Per-Class Performance Metrics (Precision, Recall, F1)",
) -> plt.Figure:
    """Plot grouped bar chart for precision, recall, and F1-score across classes.

    Args:
        per_class_data: Dictionary mapping class name/id to metrics dict (e.g. {'No DR': {'precision': 0.9, 'recall': 0.85, 'f1': 0.87}}).
        class_names: List of class labels to plot.
        save_path: Optional path to save figure.
        title: Plot title.

    Returns:
        Matplotlib Figure object.
    """
    names = class_names if class_names is not None else DEFAULT_CLASS_NAMES
    num_classes = len(names)

    precisions = []
    recalls = []
    f1s = []

    for i, name in enumerate(names):
        c_data = per_class_data.get(name, per_class_data.get(str(i), {}))
        precisions.append(c_data.get("precision", 0.0))
        recalls.append(c_data.get("recall", 0.0))
        f1s.append(c_data.get("f1", c_data.get("f1_score", 0.0)))

    x = np.arange(num_classes)
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bar1 = ax.bar(x - width, precisions, width, label="Precision", color="#2b5c8f")
    bar2 = ax.bar(x, recalls, width, label="Recall", color="#d95f02")
    bar3 = ax.bar(x + width, f1s, width, label="F1-Score", color="#7570b3")

    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Retinal Severity Grade", fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10, rotation=15)
    ax.set_ylim(0.0, 1.1)
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)

    # Add numeric labels above bars
    for bars in [bar1, bar2, bar3]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(
                    f"{height:.2f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    plt.tight_layout()

    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300, bbox_inches="tight")

    return fig
