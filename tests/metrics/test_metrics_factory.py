"""
Unit tests for the metric factory function.

This module contains unit tests verifying that `build_metrics` correctly instantiates
and maps requested metric identifiers to their respective computation functions.
"""

from __future__ import annotations  # Enables modern type hints (Python 3.7+)

from metrics.factory import (
    build_metrics,  # Factory function to build dictionary of metric functions
)
from metrics.metric_names import MetricName  # Enum defining supported metric names


def test_build_single_metric() -> None:
    """
    Verify that passing a single metric name returns a dictionary containing that metric.
    """

    # Build metric dictionary requesting only accuracy
    metrics = build_metrics(
        [
            MetricName.ACCURACY,
        ]
    )

    # Print created metrics dictionary for verification
    print(metrics)

    # Assert dictionary contains exactly one entry
    assert len(metrics) == 1

    # Assert ACCURACY key exists in the resulting dictionary
    assert (
        MetricName.ACCURACY
        in metrics
    )


def test_build_multiple_metrics() -> None:
    """
    Verify that passing multiple metric names returns a dictionary with all requested metrics.
    """

    # Build metric dictionary with accuracy, precision, recall, and F1
    metrics = build_metrics(
        [
            MetricName.ACCURACY,
            MetricName.PRECISION,
            MetricName.RECALL,
            MetricName.F1,
        ]
    )

    # Print created metrics dictionary for verification
    print(metrics)

    # Assert dictionary contains all 4 requested metrics
    assert len(metrics) == 4

    # Assert F1 key exists in the resulting dictionary
    assert (
        MetricName.F1
        in metrics
    )


def test_empty_metric_list() -> None:
    """
    Verify that passing an empty list of metric names returns an empty dictionary.
    """

    # Pass empty list to metric factory
    metrics = build_metrics(
        []
    )

    # Print empty result dictionary
    print(metrics)

    # Assert factory returns an empty dictionary
    assert metrics == {}
