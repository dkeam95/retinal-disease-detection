"""
Loss function registry.
"""

from __future__ import annotations

from collections.abc import Callable

from torch import Tensor
from torch import nn

from common.config.types import LossConfig

from losses.class_balanced_focal import (
    build_class_balanced_focal_loss,
)
from losses.cross_entropy import (
    build_cross_entropy,
)
from losses.focal import (
    build_focal_loss,
)
from losses.loss_names import (
    LossName,
)
from losses.weighted_cross_entropy import (
    build_weighted_cross_entropy,
)


LossBuilder = Callable[
    [
        LossConfig,
        Tensor | None,
    ],
    nn.Module,
]


LOSS_REGISTRY: dict[
    LossName,
    LossBuilder,
] = {
    LossName.CROSS_ENTROPY:
        build_cross_entropy,

    LossName.WEIGHTED_CROSS_ENTROPY:
        build_weighted_cross_entropy,

    LossName.FOCAL:
        build_focal_loss,

    LossName.CLASS_BALANCED_FOCAL:
        build_class_balanced_focal_loss,
}