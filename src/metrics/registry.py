"""Metric registry."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from torch import Tensor

from metrics.accuracy import compute_accuracy
from metrics.confusion_matrix import (
    compute_confusion_matrix,
)
from metrics.f1 import compute_f1
from metrics.metric_names import MetricName
from metrics.precision import compute_precision
from metrics.quadratic_weighted_kappa import (
    compute_quadratic_weighted_kappa,
)
from metrics.recall import compute_recall


MetricFunction = Callable[[Tensor, Tensor], Any]

_METRIC_REGISTRY: dict[
    MetricName,
    MetricFunction,
] = {
    MetricName.ACCURACY: compute_accuracy,
    MetricName.PRECISION: compute_precision,
    MetricName.RECALL: compute_recall,
    MetricName.F1: compute_f1,
    MetricName.QUADRATIC_WEIGHTED_KAPPA:
        compute_quadratic_weighted_kappa,
    MetricName.CONFUSION_MATRIX:
        compute_confusion_matrix,
}

def get_metric(metric_name: MetricName) -> MetricFunction:
    """Return metric function.
    
       Args:
        metric_name:
            Metric identifier.

    Returns:
        Metric function.

    Raises:
        KeyError:
            If metric is not registered.
    """

    try:
        return _METRIC_REGISTRY[metric_name]
    except KeyError as error:
        raise KeyError(
            f"Metric '{metric_name}'"
            "is not registered."
        ) from error


def list_metrics() -> list[MetricName]:
    """Return all registered metric names."""

    return sorted(_METRIC_REGISTRY.keys(), key=str)

