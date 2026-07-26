"""
Sampler builders for the dataloader module.

This module contains reusable sampler factory functions used during PyTorch DataLoader
construction to mitigate dataset class imbalance via inverse frequency sampling.
"""

from __future__ import annotations  # Enables modern type hints (Python 3.7+)

from collections import Counter     # Container for counting hashable objects
import typing                       # Type hints support for casting Sized objects

import torch                                                 # PyTorch tensor library
from torch.utils.data import Dataset, WeightedRandomSampler  # PyTorch dataset interface and sampler classes

from dataset.types import DataSample  # Typed container for individual dataset samples


def build_weighted_sampler(
    dataset: Dataset[DataSample],
) -> WeightedRandomSampler:
    """
    Build a weighted random sampler for imbalanced datasets.

    Parameters
    ----------
    dataset : Dataset[DataSample]
        Dataset used to compute class distributions for sampling.

    Returns
    -------
    WeightedRandomSampler
        Sampler balancing classes according to inverse class frequencies.
    """

    # Extract class labels from every dataset sample
    labels = [
        dataset[index].label
        for index in range(
            len(typing.cast(typing.Sized, dataset))
        )
    ]

    # Count occurrences of every class label across the dataset
    class_counts = Counter(labels)

    # Compute inverse frequency weight for every class label
    class_weights = {
        label: 1.0 / count
        for label, count in class_counts.items()
    }

    # Assign corresponding class weight to every individual sample
    sample_weights = [
        class_weights[label]
        for label in labels
    ]

    # Build weighted random sampler using computed sample weights
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )