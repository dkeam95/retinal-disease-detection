"""Preprocessing pipeline builders."""

from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2

from common.config.types import PreprocessingConfig


def build_train_pipeline(config: PreprocessingConfig) -> A.Compose:
    """Build preprocessing pipeline for training.

    Args:
        config: Preprocessing configuration.
    
    Returns:
        Albumentations pipeline for training.
    """
    return A.Compose([
        A.Resize(
            height=config.image_size,
            width=config.image_size
        ),
        A.Normalize(
            mean=config.mean,
            std=config.std
        ),
        ToTensorV2()
    ])


def build_validation_pipeline(config: PreprocessingConfig) -> A.Compose:
    """Build preprocessing pipeline for validation.

    Args:
        config: Preprocessing configuration.
    
    Returns:
        Albumentations pipeline for validation.
    """
    return A.Compose([
        A.Resize(
            height=config.image_size,
            width=config.image_size
        ),
        A.Normalize(
            mean=config.mean,
            std=config.std
        ),
        ToTensorV2()
    ])


def build_test_pipeline(config: PreprocessingConfig) -> A.Compose:
    """Build preprocessing pipeline for testing.

    Args:
        config: Preprocessing configuration.
    
    Returns:
        Albumentations pipeline for testing.
    """
    return A.Compose([
        A.Resize(
            height=config.image_size,
            width=config.image_size
        ),
        A.Normalize(
            mean=config.mean,
            std=config.std
        ),
        ToTensorV2()
    ])