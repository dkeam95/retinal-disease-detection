"""
Factory functions for PyTorch DataLoaders.

This module provides builder functions to instantiate configured PyTorch DataLoader objects
for training, validation, and testing split datasets, handling class balancing samplers,
custom batch collation, and multi-process loading settings.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from dataloader.sampler import build_weighted_sampler
from dataloader.types import DataLoaderConfig
from dataset.types import DataSample


@dataclass(frozen=True, slots=True)
class BatchSample:
    """
    Batched data container returned by collate_datasamples.
    """

    image: torch.Tensor
    label: torch.Tensor
    image_paths: list[Path]


def collate_datasamples(
    batch: list[DataSample],
    target_size: tuple[int, int] = (224, 224),
) -> BatchSample:
    """
    Collate a list of DataSample objects into a single BatchSample tensor batch.

    Parameters
    ----------
    batch : list[DataSample]
        List of samples from RetinalDataset.
    target_size : tuple[int, int], default=(512, 512)
        Target spatial dimensions (height, width) for image resizing.

    Returns
    -------
    BatchSample
        Batched images tensor (N, C, H, W), labels tensor (N,), and image paths list.
    """
    import cv2

    images: list[torch.Tensor] = []
    labels: list[int] = []
    paths: list[Path] = []

    mean_tensor = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std_tensor = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    for sample in batch:
        img = sample.image
        if isinstance(img, np.ndarray):
            # Ensure spatial dimensions match target_size (height, width)
            if (img.shape[0], img.shape[1]) != target_size:
                img = cv2.resize(
                    img, (target_size[1], target_size[0]), interpolation=cv2.INTER_AREA
                )

            # Convert HWC uint8 (0..255) to CHW float tensor (0..1)
            tensor_img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            # ImageNet channel-wise normalization
            tensor_img = (tensor_img - mean_tensor) / std_tensor
        elif isinstance(img, torch.Tensor):
            if img.ndim == 3 and (img.shape[1], img.shape[2]) != target_size:
                img = torch.nn.functional.interpolate(
                    img.unsqueeze(0),
                    size=target_size,
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
            tensor_img = img
        else:
            tensor_img = torch.tensor(img, dtype=torch.float32)

        images.append(tensor_img)
        labels.append(sample.label)
        paths.append(sample.image_path)

    stacked_images = torch.stack(images, dim=0)
    stacked_labels = torch.tensor(labels, dtype=torch.long)
    return BatchSample(image=stacked_images, label=stacked_labels, image_paths=paths)


def build_train_dataloader(
    dataset: Dataset[DataSample],
    config: DataLoaderConfig,
) -> DataLoader:
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
    DataLoader
        Configured PyTorch training dataloader.
    """
    sampler = build_weighted_sampler(dataset) if config.weight_class_balance else None

    shuffle = False if sampler is not None else config.shuffle

    return DataLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=config.drop_last,
        collate_fn=collate_datasamples,
        persistent_workers=(
            config.persistent_workers if config.num_workers > 0 else False
        ),
    )


def build_validation_dataloader(
    dataset: Dataset[DataSample],
    config: DataLoaderConfig,
) -> DataLoader:
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
    DataLoader
        Configured PyTorch validation dataloader.
    """
    return DataLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=False,
        collate_fn=collate_datasamples,
        persistent_workers=(
            config.persistent_workers if config.num_workers > 0 else False
        ),
    )


def build_test_dataloader(
    dataset: Dataset[DataSample],
    config: DataLoaderConfig,
) -> DataLoader:
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
    DataLoader
        Configured PyTorch test dataloader.
    """
    return DataLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=False,
        collate_fn=collate_datasamples,
        persistent_workers=(
            config.persistent_workers if config.num_workers > 0 else False
        ),
    )


class DataLoaderFactory:
    """Factory interface for building DataLoaders."""

    @staticmethod
    def build(
        dataset: Dataset[DataSample],
        config: DataLoaderConfig,
        training: bool = True,
    ) -> DataLoader:
        if training:
            return build_train_dataloader(dataset, config)
        return build_validation_dataloader(dataset, config)
