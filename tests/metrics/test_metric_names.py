"""
Unit tests for the MetricName enumeration.

This module contains unit tests verifying that MetricName enum values behave
as string types, allow string comparisons, and support proper lookup by string identifiers.
"""

from __future__ import annotations  # Enables modern type hints (Python 3.7+)

from metrics.metric_names import (
    MetricName,  # Enum defining supported metric name constants
)


def test_metric_names_are_strings() -> None:
    """
    Verify that MetricName enum members equal their corresponding string literals.
    """

    # Confirm ACCURACY enum value equals string 'accuracy'
    assert (
        MetricName.ACCURACY
        == "accuracy"
    )

    # Confirm F1 enum value equals string 'f1'
    assert (
        MetricName.F1
        == "f1"
    )

    # Confirm CONFUSION_MATRIX enum value equals string 'confusion_matrix'
    assert (
        MetricName.CONFUSION_MATRIX
        == "confusion_matrix"
    )


def test_metric_name_lookup() -> None:
    """
    Verify that a MetricName enum member can be instantiated from a string identifier.
    """

    # Convert string 'precision' into corresponding MetricName enum instance
    metric = MetricName(
        "precision"
    )

    # Assert string lookup yields correct PRECISION enum member
    assert (
        metric
        is MetricName.PRECISION
    )