"""
Metric factory.
"""

from __future__ import annotations

from metrics.metric_names import MetricName
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
    Build metric dictionary.

    Args:
        metric_names:
            Metrics to build.

    Returns:
        Dictionary of metric functions.
    """

    metrics: dict[
        MetricName,
        MetricFunction,
    ] = {}

    for metric_name in metric_names:

        metrics[
            metric_name
        ] = get_metric(
            metric_name,
        )

    return metrics