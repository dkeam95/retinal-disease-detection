"""
Configuration loader for the project.

This module provides functionality for loading YAML configuration
files and converting them into strongly typed configuration objects.
"""

from __future__ import annotations  # Enables modern type hints (Python 3.7+)

from pathlib import Path              # Standard library for object-oriented filesystem paths
from typing import Any                # Type hint for arbitrary dictionary values

import yaml                           # PyYAML library for parsing YAML files

from .exceptions import (
    ConfigFileNotFoundError,    # Raised when the targeted YAML file does not exist
    ConfigurationParsingError,  # Raised when YAML parsing encounters syntax errors
    InvalidConfigurationError,  # Raised when schema or mandatory fields are invalid/missing
)
from .types import (
    DatasetConfig,              # Dataclass for dataset directory and class configuration
    ExperimentConfig,           # Dataclass for experiment metadata and seeds
    LossConfig,                 # Dataclass for loss function hyperparameters
    MetricsConfig,              # Dataclass for evaluation metric configurations
    ModelConfig,                # Dataclass for neural network architecture properties
    PreprocessingConfig,        # Dataclass for image transformation and augmentation parameters
    ProjectConfig,              # Root dataclass containing all sub-configurations
    TrainingConfig,             # Dataclass for training and hardware parameters
    DataLoaderConfig,           # Dataclass for dataloader parameters
)
from .validator import validate_config  # Function that performs domain-specific validation rules


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

        # Step 1: Read raw YAML dictionary from disk
        data = ConfigLoader._read_yaml(path)

        # Step 2: Convert raw dictionary to strongly typed dataclass structures
        config = ConfigLoader._to_project_config(data)

        # Step 3: Validate logical assertions and value boundaries on the typed config
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

        # Ensure path is a Path object for unified filesystem operations
        path = Path(path)

        # Check if the configuration file exists before attempting to open
        if not path.exists():
            raise ConfigFileNotFoundError(
                f"Configuration file not found: {path}"
            )

        try:
            # Safely open and parse the YAML file using UTF-8 encoding
            with path.open("r", encoding="utf-8") as file:
                data = yaml.safe_load(file)

        except yaml.YAMLError as error:
            # Wrap PyYAML syntax/parsing errors in a domain-specific exception
            raise ConfigurationParsingError(
                "Failed to parse YAML configuration."
            ) from error

        # Guard against completely empty files
        if data is None:
            raise InvalidConfigurationError(
                "Configuration file is empty."
            )

        # Ensure the root element of the parsed YAML is a key-value dictionary
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
            # Instantiate typed dataclasses by extracting values from dictionary keys
            return ProjectConfig(
                dataset=DatasetConfig(
                    path=Path(data["dataset"]["path"]),  # Convert path string to Path instance
                    annotation_file=data["dataset"]["annotation_file"],
                    image_directory=data["dataset"]["image_directory"],
                    num_classes=data["dataset"]["num_classes"],
                ),
                preprocessing=PreprocessingConfig(
                    image_size=data["preprocessing"]["image_size"],
                    mean=tuple(data["preprocessing"]["mean"]),  # Convert list to immutable tuple
                    std=tuple(data["preprocessing"]["std"]),    # Convert list to immutable tuple
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
                dataloader=DataLoaderConfig(
                    batch_size=data["dataloader"]["batch_size"],
                    num_workers=data["dataloader"]["num_workers"],
                    shuffle=data["dataloader"]["shuffle"],
                    pin_memory=data["dataloader"]["pin_memory"],
                    drop_last=data["dataloader"]["drop_last"],
                    persistent_workers=data["dataloader"]["persistent_workers"],
                ),
                model=ModelConfig(
                    architecture=data["model"]["architecture"],
                    pretrained=data["model"]["pretrained"],
                    num_classes=data["model"]["num_classes"],
                ),
                loss=LossConfig(
                    name=data["loss"]["name"],
                    # Optional parameter extraction with fallback defaults
                    class_weights=data["loss"].get("class_weights", False),
                    gamma=data["loss"].get("gamma", 2.0),
                    alpha=data["loss"].get("alpha"),
                    beta=data["loss"].get("beta", 0.9999),
                    reduction=data["loss"].get("reduction", "mean"),
                ),
                metrics=MetricsConfig(
                    primary=tuple(data["metrics"]["primary"]),      # Convert primary metric names to tuple
                    secondary=tuple(data["metrics"]["secondary"]),  # Convert secondary metric names to tuple
                    per_class=data["metrics"].get("per_class", True),
                ),
                experiment=ExperimentConfig(
                    name=data["experiment"]["name"],
                    seed=data["experiment"]["seed"],
                ),
            )

        except (TypeError, KeyError, ValueError) as error:
            # Catch key access errors or type conversion issues and raise a clean validation exception
            raise InvalidConfigurationError(
                f"Invalid configuration: {error}"
            ) from error