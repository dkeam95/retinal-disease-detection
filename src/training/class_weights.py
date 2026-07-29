"""Class weights utility."""

from __future__ import annotations

import torch

DDR_CLASS_COUNTS: list[int] = [3486, 353, 1500, 173, 808]


def compute_class_weights(class_counts: list[int] | None = None) -> torch.Tensor:
    """Compute normalized inverse class weights."""
    counts = class_counts or DDR_CLASS_COUNTS
    total = sum(counts)
    weights = [total / (len(counts) * max(c, 1)) for c in counts]
    weights_tensor = torch.tensor(weights, dtype=torch.float32)
    return weights_tensor / weights_tensor.sum() * len(counts)
