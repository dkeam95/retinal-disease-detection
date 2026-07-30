"""Preprocessing pipeline builders.

This module builds Albumentations preprocessing pipelines for
training, validation, and testing."""

from __future__ import annotations  # Enables modern type hints (Python 3.7+)

from typing import Any  # Generic type annotation for dynamic transformation objects

import albumentations as A  # High-performance image augmentation library

from common.config.types import (
    PreprocessingConfig,  # Raw dataclass holding preprocessing parameters
)
from preprocessing.config import (
    PreprocessingSettings,  # Wrapper adapter providing property access to config
)
from preprocessing.transforms import (  # Modular transform factory functions
    brightness_contrast,
    horizontal_flip,
    normalize,
    resize,
    rotation,
    to_tensor,
    vertical_flip,
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

    # Construct deterministic base pipeline: Resize -> Normalize -> Convert to PyTorch Tensor
    return A.Compose([resize(settings), normalize(settings), to_tensor()])


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

    # Wrap raw configuration in settings adapter
    settings = PreprocessingSettings(config)

    # Initial mandatory spatial transformation
    transforms: list[Any] = [resize(settings)]

    # Probabilistic augmentations added dynamically based on configuration thresholds

    if settings.horizontal_flip_prob > 0:
        transforms.append(horizontal_flip(settings))

    if settings.vertical_flip_prob > 0:
        transforms.append(vertical_flip(settings))

    if settings.rotation_prob > 0:
        transforms.append(rotation(settings))

    if settings.brightness_contrast_prob > 0:
        transforms.append(brightness_contrast(settings))

    # Terminal normalization and PyTorch tensor conversion
    transforms.extend([normalize(settings), to_tensor()])

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

    # Build deterministic non-augmented pipeline for validation evaluation
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

    # Build deterministic non-augmented pipeline for inference/testing
    return _build_base_pipeline(settings)
