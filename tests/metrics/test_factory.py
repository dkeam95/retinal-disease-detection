import pytest

from metrics.factory import build_metrics
from metrics.metric_names import MetricName


def test_build_single_metric() -> None:
    """Verify building one metric."""

    metrics = build_metrics(
        [
            MetricName.ACCURACY,
        ]
    )

    print(metrics)

    assert len(metrics) == 1

    assert (
        MetricName.ACCURACY
        in metrics
    )


def test_build_multiple_metrics() -> None:
    """Verify building multiple metrics."""

    metrics = build_metrics(
        [
            MetricName.ACCURACY,
            MetricName.PRECISION,
            MetricName.RECALL,
            MetricName.F1,
        ]
    )

    print(metrics)

    assert len(metrics) == 4

    assert (
        MetricName.F1
        in metrics
    )


def test_empty_metric_list() -> None:
    """Verify empty configuration."""

    metrics = build_metrics(
        []
    )

    print(metrics)

    assert metrics == {}