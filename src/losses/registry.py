"""
Loss function registry.
"""

from __future__ import annotations               # Enables modern type hints (Python 3.7+)

from collections.abc import Callable             # Abstract base class for callable type annotations

from torch import Tensor                         # Type annotation for PyTorch Tensors
from torch import nn                             # PyTorch base module class

from common.config.types import LossConfig       # Configuration dataclass for loss parameters

from losses.class_balanced_focal import (        # Import builder for class-balanced focal loss
    build_class_balanced_focal_loss,
)
from losses.cross_entropy import (               # Import builder for standard cross-entropy loss
    build_cross_entropy,
)
from losses.focal import (                       # Import builder for focal loss
    build_focal_loss,
)
from losses.loss_names import (                  # Import enum defining valid loss names
    LossName,
)
from losses.weighted_cross_entropy import (      # Import builder for weighted cross-entropy loss
    build_weighted_cross_entropy,
)


LossBuilder = Callable[                          # Type alias for loss factory builder functions
    [
        LossConfig,
        Tensor | None,
    ],
    nn.Module,
]


LOSS_REGISTRY: dict[                            # Global dictionary mapping LossName enum to builders
    LossName,
    LossBuilder,
] = {
    LossName.CROSS_ENTROPY:
        build_cross_entropy,                    # Register builder for standard cross-entropy

    LossName.WEIGHTED_CROSS_ENTROPY:
        build_weighted_cross_entropy,           # Register builder for weighted cross-entropy

    LossName.FOCAL:
        build_focal_loss,                       # Register builder for focal loss

    LossName.CLASS_BALANCED_FOCAL:
        build_class_balanced_focal_loss,        # Register builder for class-balanced focal loss
}