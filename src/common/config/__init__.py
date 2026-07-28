"""
Configuration management system for Retinal Disease Detection.

This package provides utilities for loading, validating, and managing
project configurations using structured dataclasses.
"""

from .exceptions import (
    ConfigFileNotFoundError,
    ConfigurationError,
    ConfigurationParsingError,
    InvalidConfigurationError,
)
from .loader import ConfigLoader
from .types import (
    DatasetConfig,
    ExperimentConfig,
    ModelConfig,
    PreprocessingConfig,
    ProjectConfig,
    TrainingConfig,
)
from .validator import validate_config

__all__ = [
    # Primary Loader Facade
    "ConfigLoader",
    "validate_config",
    # Data Structures & Schemas
    "ProjectConfig",
    "DatasetConfig",
    "PreprocessingConfig",
    "TrainingConfig",
    "ModelConfig",
    "ExperimentConfig",
    # Custom Exceptions
    "ConfigurationError",
    "ConfigFileNotFoundError",
    "ConfigurationParsingError",
    "InvalidConfigurationError",
]