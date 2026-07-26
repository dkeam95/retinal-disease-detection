"""
Factory functions for PyTorch DataLoaders.

This module provides builder functions to instantiate configured PyTorch DataLoader objects
for training, validation, and testing split datasets, handling class balancing samplers and
multi-process loading settings.
"""

from __future__ import annotations                # Enables modern type hints (Python 3.7+)

from torch.utils.data import DataLoader, Dataset  # PyTorch dataset interface and data loader utilities

from dataloader.sampler import build_weighted_sampler  # Function to construct class-balanced weighted sampler
from dataloader.types import DataLoaderConfig          # Dataclass defining data loader runtime configurations
from dataset.types import DataSample                   # Typed container for individual dataset samples


def build_train_dataloader(
    dataset: Dataset[DataSample],
    config: DataLoaderConfig,
) -> DataLoader[DataSample]:
    """
    Build the training DataLoader.

    Parameters
    ----------
    dataset : Dataset[DataSample]
        Training dataset instance.
    config : DataLoaderConfig
        DataLoader execution configuration parameters.

    Returns
    -------
    DataLoader[DataSample]
        Configured PyTorch training dataloader.
    """

    # Use weighted sampling if requested to compensate for class imbalance
    sampler = (
        build_weighted_sampler(dataset)
        if config.weight_class_balance
        else None
    )

    # Disable shuffle when sampler is active (PyTorch requirement)
    shuffle = (
        False
        if sampler is not None
        else config.shuffle
    )

    # Instantiate PyTorch DataLoader with train-specific sampler and concurrency options
    return DataLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=config.drop_last,
        persistent_workers=(
            config.persistent_workers
            if config.num_workers > 0
            else False
        ),
    )


def build_validation_dataloader(
    dataset: Dataset[DataSample],
    config: DataLoaderConfig,
) -> DataLoader[DataSample]:
    """
    Build the validation DataLoader.

    Parameters
    ----------
    dataset : Dataset[DataSample]
        Validation dataset instance.
    config : DataLoaderConfig
        DataLoader execution configuration parameters.

    Returns
    -------
    DataLoader[DataSample]
        Configured PyTorch validation dataloader.
    """

    # Instantiate PyTorch DataLoader with evaluation options
    return DataLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=config.drop_last,
        persistent_workers=(
            config.persistent_workers
            if config.num_workers > 0
            else False
        ),
    )


def build_test_dataloader(
    dataset: Dataset[DataSample],
    config: DataLoaderConfig,
) -> DataLoader[DataSample]:
    """
    Build the test DataLoader.

    Parameters
    ----------
    dataset : Dataset[DataSample]
        Test dataset instance.
    config : DataLoaderConfig
        DataLoader execution configuration parameters.

    Returns
    -------
    DataLoader[DataSample]
        Configured PyTorch test dataloader.
    """

    # Instantiate PyTorch DataLoader with test inference options
    return DataLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=config.drop_last,
        persistent_workers=(
            config.persistent_workers
            if config.num_workers > 0
            else False
        ),
    )