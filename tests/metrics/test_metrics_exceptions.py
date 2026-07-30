"""
Unit tests for metric exceptions.

This module contains unit tests verifying that custom exception classes in the metrics package
can be raised correctly and maintain the proper inheritance hierarchy.
"""

from __future__ import annotations  # Enables modern type hints (Python 3.7+)

import pytest

from metrics.exceptions import (
    MetricError,
    MetricInitializationError,
    UnknownMetricError,
)


def test_unknown_metric_error() -> None:
    """
    Verify that UnknownMetricError can be raised and caught as expected.
    """

    # Assert that raising UnknownMetricError is correctly captured by pytest
    with pytest.raises(
        UnknownMetricError,
    ):
        raise UnknownMetricError("Unknown metric.")


def test_metric_initialization_error() -> None:
    """
    Verify that MetricInitializationError can be raised and caught as expected.
    """

    # Assert that raising MetricInitializationError is correctly captured by pytest
    with pytest.raises(
        MetricInitializationError,
    ):
        raise MetricInitializationError("Initialization failed.")


def test_metric_error_inheritance() -> None:
    """
    Verify that domain-specific metric exceptions inherit from base MetricError class.
    """

    # Assert that UnknownMetricError is a subclass of MetricError
    assert issubclass(
        UnknownMetricError,
        MetricError,
    )

    # Assert that MetricInitializationError is a subclass of MetricError
    assert issubclass(
        MetricInitializationError,
        MetricError,
    )
