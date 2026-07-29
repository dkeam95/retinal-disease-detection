"""
Common utilities shared across the project.
"""

# Import configuration management utilities, custom exceptions, and dataclass schemas
from .config import (
    ConfigFileNotFoundError,  # Raised when the specified configuration file path is missing
    ConfigLoader,  # Service/Class responsible for loading and parsing YAML configs
    ConfigurationError,  # Base exception class for all configuration-related errors
    ConfigurationParsingError,  # Raised when YAML parsing fails due to syntax errors
    DatasetConfig,  # Dataclass defining dataset-related parameters (paths, classes, etc.)
    ExperimentConfig,  # Dataclass defining experiment metadata and reproducibility seeds
    InvalidConfigurationError,  # Raised when config schema or field types fail validation
    ModelConfig,  # Dataclass defining neural network architecture and pretrained flags
    PreprocessingConfig,  # Dataclass defining data augmentation and transformation steps
    ProjectConfig,  # Root dataclass combining all sub-configurations into one schema
    TrainingConfig,  # Dataclass defining optimization and training hyperparameters
    validate_config,  # Function to validate parsed configuration schemas
)

# Explicitly define public API exports for package callers when importing via wildcard (`from common import *`)
__all__ = [
    "ConfigLoader",
    "validate_config",
    "ConfigurationError",
    "ConfigFileNotFoundError",
    "ConfigurationParsingError",
    "InvalidConfigurationError",
    "DatasetConfig",
    "ExperimentConfig",
    "ModelConfig",
    "PreprocessingConfig",
    "ProjectConfig",
    "TrainingConfig",
]
