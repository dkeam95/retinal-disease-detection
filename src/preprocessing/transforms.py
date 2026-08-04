"""Image transformation for the preprocessing module.

This module contains reusable Albumentations transformations used
to build preprocessing pipelines."""

from __future__ import annotations  # Enables modern type hints (Python 3.7+)

import albumentations as A  # High-performance image augmentation library
from albumentations.pytorch import (
    ToTensorV2,  # Convert numpy images to PyTorch Tensors (HWC -> CHW)
)

from preprocessing.config import (
    PreprocessingSettings,  # Wrapper adapter providing property access to config
)


def resize(settings: PreprocessingSettings) -> A.Resize:
    """Create a resize transformation.

    Parameters
    ----------
    settings : PreprocessingSettings
        Preprocessing configuration.

    Returns
    -------
    A.Resize
        Resize transformation.
    """

    h = (
        settings.image_size[0]
        if isinstance(settings.image_size, (tuple, list))
        else int(settings.image_size)
    )
    w = (
        settings.image_size[1]
        if isinstance(settings.image_size, (tuple, list))
        else int(settings.image_size)
    )
    return A.Resize(
        height=h,
        width=w,
    )


def horizontal_flip(settings: PreprocessingSettings) -> A.HorizontalFlip:
    """
    Create a horizontal flip transformation.

    Parameters
    ----------
    settings : PreprocessingSettings
        Preprocessing configuration.

    Returns
    -------
    A.HorizontalFlip
        Horizontal flip transformation.
    """

    # Apply random horizontal flip with target probability
    return A.HorizontalFlip(
        p=settings.horizontal_flip_prob,
    )


def vertical_flip(settings: PreprocessingSettings) -> A.VerticalFlip:
    """
    Create a vertical flip transformation.

    Parameters
    ----------
    settings : PreprocessingSettings
        Preprocessing configuration.

    Returns
    -------
    A.VerticalFlip
        Vertical flip transformation.
    """

    # Apply random vertical flip with target probability
    return A.VerticalFlip(
        p=settings.vertical_flip_prob,
    )


def rotation(settings: PreprocessingSettings) -> A.Rotate:
    """
    Create a rotation transformation.

    Parameters
    ----------
    settings : PreprocessingSettings
        Preprocessing configuration.

    Returns
    -------
    A.Rotate
        Rotation transformation.
    """

    # Rotate image within [-limit, +limit] degree range with target probability
    return A.Rotate(
        limit=settings.rotation_limit,
        p=settings.rotation_prob,
    )


def brightness_contrast(settings: PreprocessingSettings) -> A.RandomBrightnessContrast:
    """
    Create a brightness contrast transformation.

    Parameters
    ----------
    settings : PreprocessingSettings
        Preprocessing configuration.

    Returns
    -------
    A.RandomBrightnessContrast
        Brightness contrast transformation.
    """

    # Adjust image brightness and contrast randomly with target probability
    return A.RandomBrightnessContrast(
        p=settings.brightness_contrast_prob,
    )


def normalize(settings: PreprocessingSettings) -> A.Normalize:
    """
    Create a normalize transformation.

    Parameters
    ----------
    settings : PreprocessingSettings
        Preprocessing configuration.

    Returns
    -------
    A.Normalize
        Normalize transformation.
    """

    # Normalize image channels using specified mean and standard deviation: (x - mean) / std
    return A.Normalize(
        mean=settings.mean,
        std=settings.std,
    )


def to_tensor() -> A.ToTensorV2:
    """
    Create a to tensor transformation.

    Returns
    -------
    A.ToTensorV2
        To tensor transformation.
    """

    # Convert image array format from NumPy (H, W, C) to PyTorch Tensor (C, H, W)
    return ToTensorV2()
