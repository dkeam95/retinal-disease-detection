"""Configuration validator.

This module validates a fully constructed ProjectConfig object.
"""

from __future__ import annotations

from .exceptions import InvalidConfigurationError
from .types import ProjectConfig


def validate_config(config: ProjectConfig) -> None:
    """Validate project configuration across all domain layers.

    Args:
        config: Project configuration object to validate.

    Raises:
        InvalidConfigurationError: If any boundary condition or logic check fails.
    """
    _validate_dataset(config)
    _validate_preprocessing(config)
    _validate_dataloader(config)
    _validate_training(config)
    _validate_model(config)
    _validate_loss(config)
    _validate_metrics(config)
    _validate_experiment(config)


def _validate_dataset(config: ProjectConfig) -> None:
    """Validate dataset configuration."""
    if not config.dataset.path.exists():
        raise InvalidConfigurationError(
            param_name="dataset.path",
            value=config.dataset.path,
            reason=f"Dataset path does not exist on disk: {config.dataset.path}",
        )

    if config.dataset.num_classes <= 0:
        raise InvalidConfigurationError(
            param_name="dataset.num_classes",
            value=config.dataset.num_classes,
            reason="Number of classes must be greater than zero.",
        )


def _validate_preprocessing(config: ProjectConfig) -> None:
    """Validate preprocessing and augmentation settings."""
    prep = config.preprocessing

    # Validate image dimensions tuple (height, width)
    height, width = prep.image_size
    if height <= 0 or width <= 0:
        raise InvalidConfigurationError(
            param_name="preprocessing.image_size",
            value=prep.image_size,
            reason="Both height and width in image_size must be greater than zero.",
        )

    if len(prep.mean) != 3:
        raise InvalidConfigurationError(
            param_name="preprocessing.mean",
            value=prep.mean,
            reason="Normalization mean must contain exactly 3 RGB values.",
        )

    if len(prep.std) != 3:
        raise InvalidConfigurationError(
            param_name="preprocessing.std",
            value=prep.std,
            reason="Normalization std must contain exactly 3 RGB values.",
        )

    # Check probabilities range [0.0, 1.0]
    probabilities = (
        (prep.horizontal_flip_prob, "horizontal_flip_prob"),
        (prep.vertical_flip_prob, "vertical_flip_prob"),
        (prep.rotation_prob, "rotation_prob"),
        (prep.brightness_contrast_prob, "brightness_contrast_prob"),
    )

    for prob, name in probabilities:
        if not 0.0 <= prob <= 1.0:
            raise InvalidConfigurationError(
                param_name=f"preprocessing.{name}",
                value=prob,
                reason="Probability value must be between 0.0 and 1.0.",
            )

    if prep.rotation_limit < 0:
        raise InvalidConfigurationError(
            param_name="preprocessing.rotation_limit",
            value=prep.rotation_limit,
            reason="Rotation limit cannot be negative.",
        )


def _validate_dataloader(config: ProjectConfig) -> None:
    """Validate DataLoader configuration parameters."""
    dl = config.dataloader

    if dl.batch_size <= 0:
        raise InvalidConfigurationError(
            param_name="dataloader.batch_size",
            value=dl.batch_size,
            reason="Batch size must be greater than zero.",
        )

    if dl.num_workers < 0:
        raise InvalidConfigurationError(
            param_name="dataloader.num_workers",
            value=dl.num_workers,
            reason="num_workers cannot be negative.",
        )


def _validate_training(config: ProjectConfig) -> None:
    """Validate general training environment configuration."""
    training = config.training

    if training.epochs <= 0:
        raise InvalidConfigurationError(
            param_name="training.epochs",
            value=training.epochs,
            reason="Epoch count must be greater than zero.",
        )


def _validate_model(config: ProjectConfig) -> None:
    """Validate model architecture parameters against dataset bounds."""
    model = config.model
    dataset = config.dataset

    if not model.architecture.strip():
        raise InvalidConfigurationError(
            param_name="model.architecture",
            value=model.architecture,
            reason="Model architecture identifier cannot be empty.",
        )

    if model.num_classes != dataset.num_classes:
        raise InvalidConfigurationError(
            param_name="model.num_classes",
            value=model.num_classes,
            reason=f"Model output classes ({model.num_classes}) must match dataset classes ({dataset.num_classes}).",
        )


def _validate_loss(config: ProjectConfig) -> None:
    """Validate loss function parameters and loss specific hyper-parameters."""
    supported_losses = {
        "cross_entropy",
        "weighted_cross_entropy",
        "focal",
        "class_balanced_focal",
    }

    if config.loss.name not in supported_losses:
        raise InvalidConfigurationError(
            param_name="loss.name",
            value=config.loss.name,
            reason=f"Unsupported loss function. Must be one of: {supported_losses}",
        )

    if config.loss.gamma < 0:
        raise InvalidConfigurationError(
            param_name="loss.gamma",
            value=config.loss.gamma,
            reason="Focal loss gamma parameter cannot be negative.",
        )

    if not 0.0 < config.loss.beta < 1.0:
        raise InvalidConfigurationError(
            param_name="loss.beta",
            value=config.loss.beta,
            reason="Class balanced loss beta must be strictly within (0.0, 1.0).",
        )

    supported_reductions = {"mean", "sum", "none"}
    if config.loss.reduction not in supported_reductions:
        raise InvalidConfigurationError(
            param_name="loss.reduction",
            value=config.loss.reduction,
            reason=f"Loss reduction must be one of: {supported_reductions}",
        )


def _validate_metrics(config: ProjectConfig) -> None:
    """Validate target evaluation metrics."""
    supported_metrics = {
        "qwk",
        "macro_f1",
        "accuracy",
        "precision",
        "recall",
        "balanced_accuracy",
        "weighted_f1",
    }

    for metric in config.metrics.primary:
        if metric not in supported_metrics:
            raise InvalidConfigurationError(
                param_name="metrics.primary",
                value=metric,
                reason=f"Unsupported primary metric. Supported options: {supported_metrics}",
            )

    for metric in config.metrics.secondary:
        if metric not in supported_metrics:
            raise InvalidConfigurationError(
                param_name="metrics.secondary",
                value=metric,
                reason=f"Unsupported secondary metric. Supported options: {supported_metrics}",
            )


def _validate_experiment(config: ProjectConfig) -> None:
    """Validate experiment tracking metadata."""
    exp = config.experiment

    if not exp.name.strip():
        raise InvalidConfigurationError(
            param_name="experiment.name",
            value=exp.name,
            reason="Experiment name cannot be empty.",
        )

    if exp.seed < 0:
        raise InvalidConfigurationError(
            param_name="experiment.seed",
            value=exp.seed,
            reason="Random seed must be a non-negative integer.",
        )