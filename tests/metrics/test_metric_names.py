from metrics.metric_names import (
    MetricName,
)


def test_metric_names_are_strings() -> None:
    """Verify metric names behave as strings."""

    assert (
        MetricName.ACCURACY
        == "accuracy"
    )

    assert (
        MetricName.F1
        == "f1"
    )

    assert (
        MetricName.CONFUSION_MATRIX
        == "confusion_matrix"
    )


def test_metric_name_lookup() -> None:
    """Verify lookup from string."""

    metric = MetricName(
        "precision"
    )

    assert (
        metric
        is MetricName.PRECISION
    )