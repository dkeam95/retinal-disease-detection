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


class ConfigLoader:
    """
    Loads and validates project configuration files.
    """

    @staticmethod
    def load(path: str | Path) -> ProjectConfig:
        """
        Load a project configuration file.

        Parameters
        ----------
        path : str | Path
            Path to the YAML configuration file.

        Returns
        -------
        ProjectConfig
            Loaded project configuration.

        Raises
        ------
        ConfigFileNotFoundError
            If the configuration file does not exist.

        ConfigurationParsingError
            If the YAML file cannot be parsed.

        InvalidConfigurationError
            If the configuration structure is invalid.
        """
        data = ConfigLoader._read_yaml(path)
        ConfigLoader._validate(data)
        return ConfigLoader._to_project_config(data)

    @staticmethod
    def _read_yaml(path: str | Path) -> dict[str, Any]:
        """
        Read a YAML configuration file.
        """
        path = Path(path)

        if not path.exists():
            raise ConfigFileNotFoundError(f"Configuration file not found: {path}")

        try:
            with path.open("r", encoding="utf-8") as file:
                data = yaml.safe_load(file)
        except yaml.YAMLError as error:
            raise ConfigurationParsingError(
                "Failed to parse YAML configuration."
            ) from error

        if not isinstance(data, dict):
            raise InvalidConfigurationError("Configuration root must be a dictionary.")

        return data

    @staticmethod
    def _validate(data: dict[str, Any]) -> None:
        """
        Validate the root configuration structure.
        """
        required_sections = (
            "dataset",
            "preprocessing",
            "training",
            "model",
            "experiment",
        )

        missing_sections = [
            section for section in required_sections if section not in data
        ]

        if missing_sections:
            raise InvalidConfigurationError(
                f"Missing configuration sections: {', '.join(missing_sections)}"
            )

    @staticmethod
    def _to_project_config(
        data: dict[str, Any],
    ) -> ProjectConfig:
        """
        Convert a dictionary into a ProjectConfig object.
        """
        return ProjectConfig(
            dataset=DatasetConfig(**data["dataset"]),
            preprocessing=PreprocessingConfig(
                **data["preprocessing"],
            ),
            training=TrainingConfig(**data["training"]),
            model=ModelConfig(**data["model"]),
            experiment=ExperimentConfig(
                **data["experiment"],
            ),
        )
