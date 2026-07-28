"""Configuration loader for the project.

This module provides functionality for loading YAML configuration
files and converting them into strongly typed configuration objects.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .exceptions import (
    ConfigFileNotFoundError,
    ConfigurationParsingError,
    InvalidConfigurationError,
)
from .types import (
    DataLoaderConfig,
    DatasetConfig,
    ExperimentConfig,
    LossConfig,
    MetricsConfig,
    ModelConfig,
    PreprocessingConfig,
    ProjectConfig,
    TrainingConfig,
)
from .validator import validate_config


class ConfigLoader:
    """Loads and parses project configuration files."""

    @staticmethod
    def load(path: str | Path) -> ProjectConfig:
        """Load and validate a project configuration.

        Args:
            path: Path to the YAML configuration file.

        Returns:
            Validated ProjectConfig object.

        Raises:
            ConfigFileNotFoundError: If the file does not exist on disk.
            ConfigurationParsingError: If YAML contains syntax errors.
            InvalidConfigurationError: If missing keys or invalid structures exist.
        """
        data = ConfigLoader._read_yaml(path)
        config = ConfigLoader._to_project_config(data, file_path=Path(path))
        validate_config(config)
        return config

    @staticmethod
    def _read_yaml(path: str | Path) -> dict[str, Any]:
        """Read a YAML configuration file safely from disk.

        Args:
            path: Target file path.

        Returns:
            Dictionary containing parsed YAML key-value pairs.
        """
        file_path = Path(path)

        if not file_path.exists():
            raise ConfigFileNotFoundError(file_path)

        try:
            with file_path.open("r", encoding="utf-8") as file:
                data = yaml.safe_load(file)
        except yaml.YAMLError as error:
            raise ConfigurationParsingError(file_path, str(error)) from error

        if data is None:
            raise InvalidConfigurationError(
                param_name="root",
                value=None,
                reason=f"Configuration file '{file_path.name}' is empty.",
            )

        if not isinstance(data, dict):
            raise InvalidConfigurationError(
                param_name="root",
                value=type(data).__name__,
                reason=f"Configuration root in '{file_path.name}' must be a dictionary.",
            )

        return data

    @staticmethod
    def _parse_image_size(value: Any) -> tuple[int, int]:
        """Convert scalar or list image size value to a (height, width) tuple."""
        if isinstance(value, int):
            return (value, value)
        if isinstance(value, (list, tuple)) and len(value) == 2:
            return (int(value[0]), int(value[1]))
        raise ValueError(
            f"Expected int or sequence of 2 integers for image_size, got {value}"
        )

    @staticmethod
    def _to_project_config(data: dict[str, Any], file_path: Path) -> ProjectConfig:
        """Convert a raw dictionary into a strongly typed ProjectConfig object."""
        try:
            # Parse preprocessing image size flexibly
            prep_data = data["preprocessing"]
            image_size = ConfigLoader._parse_image_size(prep_data["image_size"])

            return ProjectConfig(
                dataset=DatasetConfig(
                    path=Path(data["dataset"]["path"]),
                    annotation_file=str(data["dataset"]["annotation_file"]),
                    image_directory=str(data["dataset"]["image_directory"]),
                    num_classes=int(data["dataset"]["num_classes"]),
                ),
                preprocessing=PreprocessingConfig(
                    image_size=image_size,
                    mean=tuple(prep_data["mean"]),
                    std=tuple(prep_data["std"]),
                    horizontal_flip_prob=float(prep_data["horizontal_flip_prob"]),
                    vertical_flip_prob=float(prep_data["vertical_flip_prob"]),
                    rotation_limit=int(prep_data["rotation_limit"]),
                    rotation_prob=float(prep_data["rotation_prob"]),
                    brightness_contrast_prob=float(
                        prep_data["brightness_contrast_prob"]
                    ),
                ),
                dataloader=DataLoaderConfig(
                    batch_size=int(data["dataloader"]["batch_size"]),
                    shuffle=bool(data["dataloader"]["shuffle"]),
                    num_workers=int(data["dataloader"]["num_workers"]),
                    pin_memory=bool(data["dataloader"]["pin_memory"]),
                    drop_last=bool(data["dataloader"]["drop_last"]),
                    persistent_workers=bool(data["dataloader"]["persistent_workers"]),
                    weight_class_balance=bool(
                        data["dataloader"].get("weight_class_balance", False)
                    ),
                ),
                training=TrainingConfig(
                    epochs=int(data["training"]["epochs"]),
                    device=data["training"]["device"],
                ),
                model=ModelConfig(
                    architecture=str(data["model"]["architecture"]),
                    pretrained=bool(data["model"]["pretrained"]),
                    num_classes=int(data["model"]["num_classes"]),
                ),
                loss=LossConfig(
                    name=str(data["loss"]["name"]),
                    class_weights=bool(data["loss"].get("class_weights", False)),
                    gamma=float(data["loss"].get("gamma", 2.0)),
                    alpha=data["loss"].get("alpha"),
                    beta=float(data["loss"].get("beta", 0.9999)),
                    reduction=str(data["loss"].get("reduction", "mean")),
                ),
                metrics=MetricsConfig(
                    primary=tuple(data["metrics"]["primary"]),
                    secondary=tuple(data["metrics"]["secondary"]),
                    per_class=bool(data["metrics"].get("per_class", True)),
                ),
                experiment=ExperimentConfig(
                    name=str(data["experiment"]["name"]),
                    seed=int(data["experiment"]["seed"]),
                ),
            )

        except KeyError as error:
            missing_key = error.args[0]
            raise InvalidConfigurationError(
                param_name=str(missing_key),
                value="Missing",
                reason=f"Missing required configuration key in '{file_path.name}'.",
            ) from error
        except (TypeError, ValueError) as error:
            raise InvalidConfigurationError(
                param_name="structure",
                value="Malformed",
                reason=f"Type conversion or formatting failed in '{file_path.name}': {error}",
            ) from error