import pytest

from metrics.metric_names import MetricName
from metrics.registry import (
    get_metric,
    list_metrics,
)


def test_get_accuracy_metric() -> None:
    """Verify registry returns accuracy metric."""

    metric = get_metric(
        MetricName.ACCURACY,
    )

    print(f"\nMetric = {metric.__name__}")

    assert metric.__name__ == "compute_accuracy"


def test_get_qwk_metric() -> None:
    """Verify registry returns QWK metric."""

    metric = get_metric(
        MetricName.QUADRATIC_WEIGHTED_KAPPA,
    )

    print(f"\nMetric = {metric.__name__}")

    assert (
        metric.__name__
        == "compute_quadratic_weighted_kappa"
    )


def test_list_metrics() -> None:
    """Verify all metrics are registered."""

    metrics = list_metrics()

    print("\nRegistered metrics:")

    for metric in metrics:
        print(metric)

    assert len(metrics) == 6

    assert MetricName.ACCURACY in metrics
    assert MetricName.PRECISION in metrics
    assert MetricName.RECALL in metrics
    assert MetricName.F1 in metrics
    assert MetricName.QUADRATIC_WEIGHTED_KAPPA in metrics
    assert MetricName.CONFUSION_MATRIX in metrics


def test_unknown_metric() -> None:
    """Verify unknown metric raises KeyError."""

    with pytest.raises(
        ValueError,
    ):
        MetricName(
            "unknown_metric",
        )

    print(
        "\nUnknown metric correctly detected."
    )