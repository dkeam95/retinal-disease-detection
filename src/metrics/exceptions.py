"""
Metric module exceptions.

This module defines custom domain exception types raised across the metrics package,
enabling precise error handling and graceful failures during evaluation pipeline execution.
"""

from __future__ import annotations  # Enables modern type hints (Python 3.7+)


class MetricError(Exception):
    """
    Base exception class for all metric-related errors.

    Serves as the parent exception catch-all for any failures occurring
    within metric computation, registration, or validation routines.
    """


class UnknownMetricError(MetricError):
    """
    Raised when an unrecognised or unregistered metric name is requested.

    Occurs when attempting to retrieve a metric from the registry using an invalid
    identifier or an unsupported MetricName enum value.
    """


class MetricInitializationError(MetricError):
    """
    Raised when metric evaluation inputs fail structural validation checks.

    Occurs during tensor shape validation (e.g., dimension mismatches, incorrect rank,
    or incompatible batch sizes between logits and target labels).
    """
