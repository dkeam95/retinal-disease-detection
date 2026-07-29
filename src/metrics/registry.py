"""
Metric registry module.

This module provides a centralized registry for looking up, retrieving,
and managing metric calculation functions used across model evaluation
and validation pipelines.
"""

from __future__ import annotations  # Enables modern type hints (Python 3.7+)

from collections.abc import Callable
from typing import Any

from torch import Tensor

# Import individual evaluation metric calculation functions
from metrics.accuracy import (
    compute_accuracy,
)
from metrics.confusion_matrix import (
    compute_confusion_matrix,
)
from metrics.exceptions import (
    UnknownMetricError,
)
from metrics.f1 import (
    compute_f1,
)
from metrics.metric_names import (
    MetricName,
)
from metrics.precision import (
    compute_precision,
)
from metrics.quadratic_weighted_kappa import (
    compute_quadratic_weighted_kappa,
)
from metrics.recall import (
    compute_recall,
)

# Generic type alias for metric computation callables: takes (logits, targets) -> score/matrix
MetricFunction = Callable[
    [Tensor, Tensor],
    Any,
]


# Internal lookup table mapping strongly-typed MetricName enum keys to their corresponding functions
_METRIC_REGISTRY: dict[
    MetricName,
    MetricFunction,
] = {
    MetricName.ACCURACY:
        compute_accuracy,

    MetricName.PRECISION:
        compute_precision,

    MetricName.RECALL:
        compute_recall,

    MetricName.F1:
        compute_f1,

    MetricName.QUADRATIC_WEIGHTED_KAPPA:
        compute_quadratic_weighted_kappa,

    MetricName.CONFUSION_MATRIX:
        compute_confusion_matrix,
}


def get_metric(
    metric_name: MetricName,
) -> MetricFunction:
    """
    Retrieve a metric function by its enum identifier from the registry.

    Args:
        metric_name:
            Strongly-typed metric identifier (MetricName enum).

    Returns:
        MetricFunction:
            The callable metric calculation function taking (logits, targets).

    Raises:
        UnknownMetricError:
            If the requested metric identifier is not found in the registry.
    """

    try:
        # Fetch the metric computation function from the private lookup dictionary
        return _METRIC_REGISTRY[
            metric_name
        ]

    except KeyError as error:
        # Format list of available valid metrics for informative error messaging
        available_metrics = ", ".join(
            metric.value
            for metric in MetricName
        )

        raise UnknownMetricError(
            f"Unknown metric: "
            f"{metric_name!r}. "
            f"Available metrics: "
            f"{available_metrics}"
        ) from error


def list_metrics() -> list[MetricName]:
    """
    List all currently registered metric identifiers.

    Returns:
        list[MetricName]:
            A sorted list of all available registered MetricName enum members.
    """

    # Return sorted metric names lexicographically by their string representation
    return sorted(
        _METRIC_REGISTRY.keys(),
        key=str,
    )
