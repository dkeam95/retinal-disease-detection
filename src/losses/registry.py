"""Loss function registry."""

from __future__ import annotations

from collections.abc import Callable

from torch import Tensor
from torch import nn

from common.config.types import LossConfig

from .class_balanced_focal import (
    build_class_balanced_focal_loss
)
from .cross_entropy import (
    build_cross_entropy
)
from .focal import (
    build_focal_loss
)
from .weighted_cross_entropy import (
    build_weighted_cross_entropy
)


LossBuilder = Callable[
    [LossConfig, Tensor | None],
    nn.Module
]


LOSS_REGISTRY: dict[str, LossBuilder] = {
    "cross_entropy": build_cross_entropy,
    "weighted_cross_entropy": build_weighted_cross_entropy,
    "focal": build_focal_loss,
    "class_balanced_focal": build_class_balanced_focal_loss,
}

