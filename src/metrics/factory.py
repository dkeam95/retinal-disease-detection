"""
Metric factory module.

This module provides factory functions to construct and instantiate dictionaries
mapping metric identifiers to their corresponding calculation functions.
"""

from __future__ import annotations  # Enables modern type hints (Python 3.7+)

from metrics.metric_names import (
    MetricName,
)
from metrics.registry import (
    MetricFunction,
    get_metric,
)


def build_metrics(
    metric_names: list[MetricName],
) -> dict[
    MetricName,
    MetricFunction,
]:
    """
    Build a dictionary mapping metric identifiers to metric functions.

    Args:
        metric_names:
            List of requested metric enum identifiers to build.

    Returns:
        dict[MetricName, MetricFunction]:
            Dictionary mapping each requested MetricName to its corresponding
            registered MetricFunction callable.

    Raises:
        UnknownMetricError:
            If any requested metric name is not found in the registry.
    """

    # Initialize empty registry mapping for constructed metrics
    metrics: dict[
        MetricName,
        MetricFunction,
    ] = {}

    # Look up and associate each metric name with its registered implementation
    for metric_name in metric_names:
        metrics[metric_name] = get_metric(
            metric_name,
        )

    return metrics
