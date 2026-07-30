"""Unit tests for ErrorAnalyzer."""

from __future__ import annotations

import numpy as np
import pytest

from evaluation.error_analysis import ErrorAnalyzer


def test_error_analyzer(tmp_path):
    analyzer = ErrorAnalyzer()
    y_true = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
    y_pred = np.array([0, 1, 2, 3, 4, 0, 1, 2, 2, 4])

    cm = analyzer.compute_confusion_matrix(y_true, y_pred)
    assert cm.shape == (5, 5)

    summary = analyzer.generate_full_error_report(
        y_true, y_pred, output_dir=tmp_path, experiment_name="test_run"
    )
    assert "confusion_matrix" in summary
    assert (tmp_path / "test_run" / "confusion_matrix.png").exists()
    assert (tmp_path / "test_run" / "error_report.json").exists()
