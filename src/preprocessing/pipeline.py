"""Preprocessing pipeline builders.

This module builds Albumentations preprocessing pipelines for
training, validation, and testing."""

from __future__ import annotations

from typing import Any

import albumentations as A

from common.config.types import PreprocessingConfig

from preprocessing.config import PreprocessingSettings
from preprocessing.transforms import (
    brightness_contrast,
    horizontal_flip,
    normalize,
    resize,
    rotation,
    to_tensor,
    vertical_flip
)


def _build_base_pipeline(settings: PreprocessingSettings) -> A.Compose:
    """
    Build the common preprocessing pipeline.

    Parameters
    ----------
    settings : PreprocessingSettings
        Preprocessing configuration.

    Returns
    -------
    A.Compose
        Base preprocessing pipeline.
    """

    return A.Compose([
        resize(settings),
        normalize(settings),
        to_tensor()
    ])


def build_train_pipeline(config: PreprocessingConfig) -> A.Compose:
    """
    Build the preprocessing pipeline for training.

    Parameters
    ----------
    config : PreprocessingConfig
        Preprocessing configuration.

    Returns
    -------
    A.Compose
        Training preprocessing pipeline.
    """

    settings = PreprocessingSettings(config)

    transforms: list[Any] = [
        resize(settings)
    ]

    # Probalistic transformations

    if settings.horizontal_flip_prob > 0:
        transforms.append(horizontal_flip(settings))

    if settings.vertical_flip_prob > 0:
        transforms.append(vertical_flip(settings))

    if settings.rotation_prob > 0:
        transforms.append(rotation(settings))

    if settings.brightness_contrast_prob > 0:
        transforms.append(brightness_contrast(settings))

    transforms.extend([
        normalize(settings),
        to_tensor()
    ])

    return A.Compose(transforms)


def build_validation_pipeline(config: PreprocessingConfig) -> A.Compose:
    """
    Build the preprocessing pipeline for validation.

    Parameters
    ----------
    config : PreprocessingConfig
        Preprocessing configuration.

    Returns
    -------
    A.Compose
        Validation preprocessing pipeline.
    """

    settings = PreprocessingSettings(config)

    return _build_base_pipeline(settings)


def build_test_pipeline(config: PreprocessingConfig) -> A.Compose:
    """
    Build the preprocessing pipeline for testing.

    Parameters
    ----------
    config : PreprocessingConfig
        Preprocessing configuration.

    Returns
    -------
    A.Compose
        Test preprocessing pipeline.
    """

    settings = PreprocessingSettings(config)

    return _build_base_pipeline(settings)
    