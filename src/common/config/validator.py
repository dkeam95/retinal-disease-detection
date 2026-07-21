"""
Configuration validator.

This module validates a fully constructed ProjectConfig object.
"""

from __future__ import annotations  # Enables modern type hints (Python 3.7+)

from common.config.exceptions import InvalidConfigurationError  # Custom exception for failed config assertions
from common.config.types import ProjectConfig  # Strongly typed root configuration object


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

    # Run validation checks across all sub-configuration domains
    _validate_dataset(config)
    _validate_preprocessing(config)
    _validate_training(config)
    _validate_model(config)
    _validate_loss(config)
    _validate_metrics(config)
    _validate_experiment(config)


def _validate_dataset(config: ProjectConfig) -> None:
    """Validate dataset configuration."""

    # Ensure dataset path actually exists on the filesystem
    if not config.dataset.path.exists():
        raise InvalidConfigurationError(
            f"Dataset path does not exist: {config.dataset.path}"
        )

    # Class count must be a positive integer
    if config.dataset.num_classes <= 0:
        raise InvalidConfigurationError(
            "Dataset num_classes must be greater than zero."
        )


def _validate_preprocessing(config: ProjectConfig) -> None:
    """Validate preprocessing configuration."""

    preprocessing = config.preprocessing

    # Image dimensions must be strictly positive
    if preprocessing.image_size <= 0:
        raise InvalidConfigurationError(
            "Image size must be greater than zero."
        )

    # RGB image normalization requires exactly 3 mean channels
    if len(preprocessing.mean) != 3:
        raise InvalidConfigurationError(
            "Normalization mean must contain exactly 3 values."
        )

    # RGB image normalization requires exactly 3 std channels
    if len(preprocessing.std) != 3:
        raise InvalidConfigurationError(
            "Normalization std must contain exactly 3 values."
        )

    # Group probability parameters to validate [0.0, 1.0] range in a loop
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

    # Validate that probability values are within valid bound [0, 1]
    for probability, name in probability_fields:
        if not 0.0 <= probability <= 1.0:
            raise InvalidConfigurationError(
                f"{name} must be between 0.0 and 1.0."
            )

    # Rotation angle threshold cannot be negative
    if preprocessing.rotation_limit < 0:
        raise InvalidConfigurationError(
            "Rotation limit cannot be negative."
        )


def _validate_training(config: ProjectConfig) -> None:
    """Validate training configuration."""

    training = config.training

    # Batch size must be positive
    if training.batch_size <= 0:
        raise InvalidConfigurationError(
            "Batch size must be greater than zero."
        )

    # Epoch count must be positive
    if training.epochs <= 0:
        raise InvalidConfigurationError(
            "Epochs must be greater than zero."
        )

    # DataLoader multiprocessing workers cannot be negative
    if training.num_workers < 0:
        raise InvalidConfigurationError(
            "num_workers cannot be negative."
        )


def _validate_model(config: ProjectConfig) -> None:
    """Validate model configuration."""

    model = config.model
    dataset = config.dataset

    # Ensure model architecture string is non-empty after trimming whitespace
    if not model.architecture.strip():
        raise InvalidConfigurationError(
            "Model architecture cannot be empty."
        )

    # Model output layer units must match dataset target class count
    if model.num_classes != dataset.num_classes:
        raise InvalidConfigurationError(
            "Model num_classes must match dataset num_classes."
        )


def _validate_loss(config: ProjectConfig) -> None:
    """Validate loss configuration."""

    # Set of supported loss implementation identifiers
    supported_losses = {
        "cross_entropy",
        "weighted_cross_entropy",
        "focal",
        "class_balanced_focal"
    }

    # Verify that requested loss function is supported
    if config.loss.name not in supported_losses:
        raise InvalidConfigurationError(
            f"Unsupported loss function: {config.loss.name}"
        )

    # Gamma parameter for Focal Loss must be non-negative
    if config.loss.gamma < 0:
        raise InvalidConfigurationError(
            "Loss gamma must be non-negative."
        )

    # Beta hyperparameter for effective number of samples must lie strictly in (0, 1)
    if config.loss.beta <= 0 or config.loss.beta >= 1:
        raise InvalidConfigurationError(
            "Loss beta must be in the interval (0, 1)."
        )

    # Validate reduction method against supported PyTorch options
    if config.loss.reduction not in {
        "mean",
        "sum",
        "none"
    }:
        raise InvalidConfigurationError(
            "Loss reduction must be 'mean', 'sum', or 'none'."
        )


def _validate_metrics(config: ProjectConfig) -> None:
    """Validate metrics configuration."""

    # Set of supported evaluation metrics
    supported_metrics = {
        "qwk",
        "macro_f1",
        "accuracy",
        "precision",
        "recall",
        "balanced_accuracy",
        "weighted_f1"
    }

    # Verify primary metric names against supported set
    for metric in config.metrics.primary:
        if metric not in supported_metrics:
            raise InvalidConfigurationError(
                f"Unsupported primary metric: {metric}"
            )

    # Verify secondary metric names against supported set
    for metric in config.metrics.secondary:
        if metric not in supported_metrics:
            raise InvalidConfigurationError(
                f"Unsupported secondary metric: {metric}"
            )


def _validate_experiment(config: ProjectConfig) -> None:
    """Validate experiment configuration."""

    # Experiment name identifier must be non-empty
    if not config.experiment.name:
        raise InvalidConfigurationError(
            "Experiment name must not be empty."
        )

    # Random seed must be a non-negative integer
    if config.experiment.seed < 0:
        raise InvalidConfigurationError(
            "Experiment seed must be non-negative."
        )