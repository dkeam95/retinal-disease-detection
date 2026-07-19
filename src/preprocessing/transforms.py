"""Image transformation for the preprocessing module.

This module contains reusable Albumentations transformations used 
to build preprocessing pipelines."""

from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2

from preprocessing.config import PreprocessingSettings


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

    return A.Resize(
        height=settings.image_size,
        width=settings.image_size,
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

    return A.ToTensorV2()