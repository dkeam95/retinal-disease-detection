"""Metric name definitions.

This module is responsible for defining metric names.
"""

from __future__ import annotations  # Enables modern type hints (Python 3.7+)

from enum import StrEnum  # String enum base class (Python 3.11+)


class MetricName(StrEnum):
    """Supported evaluation metrics."""

    # String identifiers for model evaluation and validation metrics
    ACCURACY = "accuracy"                                # Overall prediction accuracy
    PRECISION = "precision"                              # Precision / Positive Predictive Value
    RECALL = "recall"                                    # Recall / Sensitivity / True Positive Rate
    F1 = "f1"                                            # Harmonic mean of Precision and Recall
    QUADRATIC_WEIGHTED_KAPPA = "quadratic_weighted_kappa" # Cohen's Quadratic Weighted Kappa (QWK) for ordinal classification
    CONFUSION_MATRIX = "confusion_matrix"                # Class-wise multi-class confusion matrix