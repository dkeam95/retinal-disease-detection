"""Unit tests for visualization plotting and report generator modules."""

from __future__ import annotations

import numpy as np

from visualization.plots import (
    plot_confusion_matrix,
    plot_learning_curves,
    plot_per_class_metrics,
)
from visualization.report_generator import HTMLReportGenerator


def test_plot_learning_curves(tmp_path):
    history = {
        "train_loss": [0.8, 0.5, 0.3],
        "val_loss": [0.7, 0.4, 0.25],
        "val_qwk": [0.6, 0.75, 0.82],
        "val_accuracy": [0.65, 0.78, 0.85],
    }
    save_file = tmp_path / "learning_curves.png"
    fig = plot_learning_curves(history, save_path=save_file)
    assert fig is not None
    assert save_file.exists()
    assert save_file.stat().st_size > 0


def test_plot_confusion_matrix(tmp_path):
    cm = np.array([
        [50, 2, 0, 0, 0],
        [3, 40, 5, 0, 0],
        [0, 4, 30, 2, 0],
        [0, 0, 3, 20, 1],
        [0, 0, 0, 2, 10],
    ])
    save_file = tmp_path / "cm.png"
    fig = plot_confusion_matrix(cm, save_path=save_file, normalize=True)
    assert fig is not None
    assert save_file.exists()
    assert save_file.stat().st_size > 0


def test_plot_per_class_metrics(tmp_path):
    per_class_data = {
        "No DR": {"precision": 0.9, "recall": 0.95, "f1": 0.92},
        "Mild NPDR": {"precision": 0.8, "recall": 0.75, "f1": 0.77},
        "Moderate NPDR": {"precision": 0.85, "recall": 0.82, "f1": 0.83},
        "Severe NPDR": {"precision": 0.78, "recall": 0.70, "f1": 0.74},
        "Proliferative DR": {"precision": 0.88, "recall": 0.85, "f1": 0.86},
    }
    save_file = tmp_path / "per_class.png"
    fig = plot_per_class_metrics(per_class_data, save_path=save_file)
    assert fig is not None
    assert save_file.exists()


def test_html_report_generator(tmp_path):
    history = {
        "train_loss": [0.8, 0.5],
        "val_loss": [0.7, 0.4],
        "val_qwk": [0.6, 0.8],
    }
    cm = np.eye(5, dtype=int) * 10
    per_class_metrics = {
        "No DR": {"precision": 0.9, "recall": 0.9, "f1": 0.9},
    }
    metrics = {"qwk": 0.8, "accuracy": 0.85, "macro_f1": 0.82}
    config_summary = {"Architecture": "efficientnet_b0", "Epochs": "10"}
    output_html = tmp_path / "report.html"

    report_path = HTMLReportGenerator.generate(
        experiment_name="test_exp",
        metrics=metrics,
        config_summary=config_summary,
        history=history,
        confusion_matrix=cm,
        per_class_metrics=per_class_metrics,
        output_path=output_html,
    )

    assert report_path.exists()
    assert report_path.stat().st_size > 0
    content = report_path.read_text(encoding="utf-8")
    assert "test_exp" in content
    assert "Quadratic Weighted Kappa" in content
