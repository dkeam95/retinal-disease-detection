"""
Configuration system.
"""

# Core loader and validation functions
from .loader import ConfigLoader        # Class responsible for loading configuration files from disk
from .validator import validate_config  # Function that validates configuration dictionary against schema

# Custom exceptions for configuration-related errors
from .exceptions import (
    ConfigurationError,          # Base exception class for configuration module issues
    ConfigFileNotFoundError,     # Raised when the targeted configuration file does not exist
    ConfigurationParsingError,   # Raised when parsing YAML/JSON content fails
    InvalidConfigurationError,   # Raised when configuration values fail validation rules
)

# Dataclass types defining configuration schema structures
from .types import (
    ProjectConfig,               # Root configuration dataclass aggregating all sub-configs
    DatasetConfig,               # Configuration schema for dataset paths and attributes
    PreprocessingConfig,         # Configuration schema for data transformations and augmentations
    TrainingConfig,              # Configuration schema for training hyperparameters and optimizers
    ModelConfig,                 # Configuration schema for neural network architecture selection
    ExperimentConfig,            # Configuration schema for experiment tracking and random seeds
)

# Explicitly define public API exports for module imports
__all__ = [
    "ConfigLoader",
    "validate_config",
    "ProjectConfig",
    "DatasetConfig",
    "PreprocessingConfig",
    "TrainingConfig",
    "ModelConfig",
    "ExperimentConfig",
    "ConfigurationError",
    "ConfigFileNotFoundError",
    "ConfigurationParsingError",
    "InvalidConfigurationError",
]