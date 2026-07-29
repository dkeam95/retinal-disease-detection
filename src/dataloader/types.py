"""
Type definitions and data structures for the dataloader module.

This module defines shared type aliases, dataclass configurations, and data containers
for managing PyTorch DataLoaders across training, validation, and testing pipelines.
"""

from __future__ import annotations  # Enables modern type hints (Python 3.7+)

from dataclasses import dataclass  # Decorator for creating immutable data classes

from torch.utils.data import DataLoader  # PyTorch data loading utility class

from dataset.types import DataSample  # Typed container for individual dataset samples

# Type alias for PyTorch DataLoader returning batches of DataSample instances
SampleDataLoader = DataLoader[DataSample]


@dataclass(frozen=True, slots=True)
class DataLoaderCollection:
    """
    Container aggregating DataLoaders for all dataset splits.

    Attributes
    ----------
    train : SampleDataLoader
        DataLoader instance configured for the training set.
    val : SampleDataLoader
        DataLoader instance configured for the validation set.
    test : SampleDataLoader
        DataLoader instance configured for the test set.
    """

    train: SampleDataLoader
    val: SampleDataLoader
    test: SampleDataLoader


@dataclass(frozen=True, slots=True)
class DataLoaderConfig:
    """
    Configuration parameters for constructing PyTorch DataLoaders.

    Attributes
    ----------
    batch_size : int
        Number of samples per batch.
    shuffle : bool
        Whether to reshuffle dataset samples at every epoch.
    num_workers : int
        Number of subprocesses used for data loading.
    pin_memory : bool
        If True, copies tensors into CUDA pinned memory before returning.
    drop_last : bool
        If True, drops the last incomplete batch if its size is smaller than batch_size.
    persistent_workers : bool
        If True, keeps data loader worker processes alive between epochs.
    weight_class_balance : bool
        If True, applies class-weighted sampling to mitigate dataset imbalance.
    """

    batch_size: int
    shuffle: bool
    num_workers: int
    pin_memory: bool
    drop_last: bool
    persistent_workers: bool
    weight_class_balance: bool
