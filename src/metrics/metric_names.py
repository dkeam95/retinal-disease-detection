"""Metric name definitions."""

from __future__ import annotations

from enum import StrEnum


class MetricName(StrEnum):
    """Supported evaluation metrics."""

    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1 = "f1"
    QUADRATIC_WEIGHTED_KAPPA = "quadratic_weighted_kappa"
    CONFUSION_MATRIX = "confusion_matrix"
    
    