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
                    **data["dataset"],
                ),
                preprocessing=PreprocessingConfig(
                    **data["preprocessing"],
                ),
                training=TrainingConfig(
                    **data["training"],
                ),
                model=ModelConfig(
                    **data["model"],
                ),
                experiment=ExperimentConfig(
                    **data["experiment"],
                ),
            )

        except (TypeError, KeyError) as error:
            raise InvalidConfigurationError(
                f"Invalid configuration: {error}"
            ) from error