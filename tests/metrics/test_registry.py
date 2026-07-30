"""
Unit tests for the metric registry module.

This module contains unit tests verifying that `get_metric` retrieves the expected evaluation
metric functions by their enum identifier, `list_metrics` returns the complete set of supported
metric names, and invalid metric names properly raise lookup errors.
"""

from __future__ import annotations  # Enables modern type hints (Python 3.7+)

import pytest

from metrics.metric_names import MetricName
from metrics.registry import (
    get_metric,
    list_metrics,
)


def test_get_accuracy_metric() -> None:
    """
    Verify that the registry returns the correct Accuracy computation function.
    """

    # Retrieve accuracy computation function from registry
    metric = get_metric(
        MetricName.ACCURACY,
    )

    print(f"\nMetric = {metric.__name__}")

    # Assert retrieved function name matches expected implementation
    assert metric.__name__ == "compute_accuracy"


def test_get_qwk_metric() -> None:
    """
    Verify that the registry returns the correct Quadratic Weighted Kappa computation function.
    """

    # Retrieve QWK computation function from registry
    metric = get_metric(
        MetricName.QUADRATIC_WEIGHTED_KAPPA,
    )

    print(f"\nMetric = {metric.__name__}")

    # Assert retrieved function name matches expected implementation
    assert metric.__name__ == "compute_quadratic_weighted_kappa"


def test_list_metrics() -> None:
    """
    Verify that `list_metrics` returns all 6 supported evaluation metric members.
    """

    # Retrieve list of all registered metric names
    metrics = list_metrics()

    print("\nRegistered metrics:")

    for metric in metrics:
        print(metric)

    # Assert total count of registered metrics is exactly 6
    assert len(metrics) == 6

    # Assert all required metric keys are present in the list
    assert MetricName.ACCURACY in metrics
    assert MetricName.PRECISION in metrics
    assert MetricName.RECALL in metrics
    assert MetricName.F1 in metrics
    assert MetricName.QUADRATIC_WEIGHTED_KAPPA in metrics
    assert MetricName.CONFUSION_MATRIX in metrics


def test_unknown_metric_name() -> None:
    """
    Verify that attempting to create a MetricName from an unknown string raises a ValueError.
    """

    # Assert ValueError is raised when instantiating MetricName with an unsupported string
    with pytest.raises(
        ValueError,
        match="unknown_metric",
    ):
        MetricName(
            "unknown_metric",
        )

    print("\nUnknown metric correctly detected.")
