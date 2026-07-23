"""Metric registry."""

from __future__ import annotations  # Enables modern type hints (Python 3.7+)

from collections.abc import Callable  # Type annotation for callable objects/functions
from typing import Any  # Type hint for arbitrary return values

from torch import Tensor  # Type annotation for PyTorch multi-dimensional arrays

# Import evaluation functions for all supported metrics
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

# Type alias defining standard signature for metric functions: (logits, targets) -> result
MetricFunction = Callable[[Tensor, Tensor], Any]

# Central dispatch table mapping MetricName enum keys to their implementation functions
_METRIC_REGISTRY: dict[
    MetricName,
    MetricFunction,
] = {
    MetricName.ACCURACY: compute_accuracy,
    MetricName.PRECISION: compute_precision,
    MetricName.RECALL: compute_recall,
    MetricName.F1: compute_f1,
    MetricName.QUADRATIC_WEIGHTED_KAPPA: (
        compute_quadratic_weighted_kappa
    ),
    MetricName.CONFUSION_MATRIX: (
        compute_confusion_matrix
    ),
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

    # Look up requested metric in dispatch registry, raising descriptive error if missing
    try:
        return _METRIC_REGISTRY[metric_name]
    except KeyError as error:
        raise KeyError(
            f"Metric '{metric_name}' "
            "is not registered."
        ) from error


def list_metrics() -> list[MetricName]:
    """Return all registered metric names."""

    # Return lexicographically sorted list of all available metric identifiers
    return sorted(_METRIC_REGISTRY.keys(), key=str)