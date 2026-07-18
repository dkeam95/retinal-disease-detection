"""
Configuration validator.

This module validates a fully constructed ProjectConfig object.
"""

from __future__ import annotations

from common.config.exceptions import InvalidConfigurationError
from common.config.types import ProjectConfig


def validate_config(config: ProjectConfig) -> None:
    """
    Validate project configuration.

    Parameters
    ----------
    config : ProjectConfig
        Project configuration to validate.

    Raises
    ------
    InvalidConfigurationError
        If the configuration is invalid.
    """

    _validate_dataset(config)
    _validate_preprocessing(config)
    _validate_training(config)
    _validate_model(config)


def _validate_dataset(config: ProjectConfig) -> None:
    """Validate dataset configuration."""

    if not config.dataset.path.exists():
        raise InvalidConfigurationError(
            f"Dataset path does not exist: {config.dataset.path}"
        )

    if config.dataset.num_classes <= 0:
        raise InvalidConfigurationError(
            "Dataset num_classes must be greater than zero."
        )


def _validate_preprocessing(config: ProjectConfig) -> None:
    """Validate preprocessing configuration."""

    preprocessing = config.preprocessing

    if preprocessing.image_size <= 0:
        raise InvalidConfigurationError(
            "Image size must be greater than zero."
        )

    if len(preprocessing.mean) != 3:
        raise InvalidConfigurationError(
            "Normalization mean must contain exactly 3 values."
        )

    if len(preprocessing.std) != 3:
        raise InvalidConfigurationError(
            "Normalization std must contain exactly 3 values."
        )

    probability_fields = (
        (
            preprocessing.horizontal_flip_prob,
            "horizontal_flip_prob",
        ),
        (
            preprocessing.vertical_flip_prob,
            "vertical_flip_prob",
        ),
        (
            preprocessing.brightness_contrast_prob,
            "brightness_contrast_prob",
        ),
    )

    for probability, name in probability_fields:
        if not 0.0 <= probability <= 1.0:
            raise InvalidConfigurationError(
                f"{name} must be between 0.0 and 1.0."
            )

    if preprocessing.rotation_limit < 0:
        raise InvalidConfigurationError(
            "Rotation limit cannot be negative."
        )


def _validate_training(config: ProjectConfig) -> None:
    """Validate training configuration."""

    training = config.training

    if training.batch_size <= 0:
        raise InvalidConfigurationError(
            "Batch size must be greater than zero."
        )

    if training.epochs <= 0:
        raise InvalidConfigurationError(
            "Epochs must be greater than zero."
        )

    if training.num_workers < 0:
        raise InvalidConfigurationError(
            "num_workers cannot be negative."
        )


def _validate_model(config: ProjectConfig) -> None:
    """Validate model configuration."""

    model = config.model
    dataset = config.dataset

    if not model.architecture.strip():
        raise InvalidConfigurationError(
            "Model architecture cannot be empty."
        )

    if model.num_classes != dataset.num_classes:
        raise InvalidConfigurationError(
            "Model num_classes must match dataset num_classes."
        )