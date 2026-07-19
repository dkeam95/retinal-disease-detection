"""
Configuration loader for the project.

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
    DatasetConfig,
    ExperimentConfig,
    ModelConfig,
    PreprocessingConfig,
    ProjectConfig,
    TrainingConfig,
)
from .validator import validate_config


class ConfigLoader:
    """Loads project configuration files."""

    @staticmethod
    def load(path: str | Path) -> ProjectConfig:
        """
        Load and validate a project configuration.

        Parameters
        ----------
        path : str | Path
            Path to the YAML configuration file.

        Returns
        -------
        ProjectConfig
            Validated project configuration.

        Raises
        ------
        ConfigFileNotFoundError
            If the configuration file does not exist.

        ConfigurationParsingError
            If the YAML file cannot be parsed.

        InvalidConfigurationError
            If the configuration is invalid.
        """

        data = ConfigLoader._read_yaml(path)

        config = ConfigLoader._to_project_config(data)

        validate_config(config)

        return config

    @staticmethod
    def _read_yaml(path: str | Path) -> dict[str, Any]:
        """
        Read a YAML configuration file.

        Parameters
        ----------
        path : str | Path
            Path to configuration file.

        Returns
        -------
        dict[str, Any]
            Parsed YAML content.
        """

        path = Path(path)

        if not path.exists():
            raise ConfigFileNotFoundError(
                f"Configuration file not found: {path}"
            )

        try:
            with path.open("r", encoding="utf-8") as file:
                data = yaml.safe_load(file)

        except yaml.YAMLError as error:
            raise ConfigurationParsingError(
                "Failed to parse YAML configuration."
            ) from error

        if data is None:
            raise InvalidConfigurationError(
                "Configuration file is empty."
            )

        if not isinstance(data, dict):
            raise InvalidConfigurationError(
                "Configuration root must be a dictionary."
            )

        return data

    @staticmethod
    def _to_project_config(
        data: dict[str, Any],
    ) -> ProjectConfig:
        """
        Convert a dictionary into a ProjectConfig object.

        Parameters
        ----------
        data : dict[str, Any]
            Parsed YAML dictionary.

        Returns
        -------
        ProjectConfig
            Strongly typed project configuration.
        """

        try:
            return ProjectConfig(
                dataset=DatasetConfig(
                    path=Path(data["dataset"]["path"]),
                    annotation_file=data["dataset"]["annotation_file"],
                    image_directory=data["dataset"]["image_directory"],
                    num_classes=data["dataset"]["num_classes"],
                ),
                preprocessing=PreprocessingConfig(
                    image_size=data["preprocessing"]["image_size"],
                    mean=tuple(data["preprocessing"]["mean"]),
                    std=tuple(data["preprocessing"]["std"]),
                    horizontal_flip_prob=data["preprocessing"]["horizontal_flip_prob"],
                    vertical_flip_prob=data["preprocessing"]["vertical_flip_prob"],
                    rotation_limit=data["preprocessing"]["rotation_limit"],
                    rotation_prob=data["preprocessing"]["rotation_prob"],
                    brightness_contrast_prob=data["preprocessing"]["brightness_contrast_prob"],
                ),
                training=TrainingConfig(
                    batch_size=data["training"]["batch_size"],
                    epochs=data["training"]["epochs"],
                    num_workers=data["training"]["num_workers"],
                    device=data["training"]["device"],
                ),
                model=ModelConfig(
                    architecture=data["model"]["architecture"],
                    pretrained=data["model"]["pretrained"],
                    num_classes=data["model"]["num_classes"],
                ),
                experiment=ExperimentConfig(
                    name=data["experiment"]["name"],
                    seed=data["experiment"]["seed"],
                ),
            )

        except (TypeError, KeyError, ValueError) as error:
            raise InvalidConfigurationError(
                f"Invalid configuration: {error}"
            ) from error