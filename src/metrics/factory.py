"""
Metric factory.
"""

from __future__ import annotations  # Enables modern type hints (Python 3.7+)

from metrics.metric_names import MetricName  # Strongly typed enum/type for metric identifiers
from metrics.registry import (
    MetricFunction,  # Type alias for metric function signatures
    get_metric,       # Lookup function to fetch metric implementation from registry
)


def build_metrics(
    metric_names: list[MetricName],
) -> dict[
    MetricName,
    MetricFunction,
]:
    """
    Build metric dictionary.

    Args:
        metric_names:
            Metrics to build.

    Returns:
        Dictionary of metric functions.
    """

    # Initialize empty registry map for active metric instances
    metrics: dict[
        MetricName,
        MetricFunction,
    ] = {}

    # Iterate through requested metric names and retrieve corresponding functions from registry
    for metric_name in metric_names:

        metrics[
            metric_name
        ] = get_metric(
            metric_name,
        )

    # Return configured dictionary mapping metric names to executable functions
    return metrics