"""
Unit tests for the configuration loader module.

This module contains unit tests verifying that `ConfigLoader` correctly parses valid YAML
configuration files into `ProjectConfig` instances, and properly raises domain-specific
exceptions on missing files, malformed syntax, non-dictionary root structures, or missing required sections.
"""

from __future__ import annotations  # Enables modern type hints (Python 3.7+)

from pathlib import Path  # Object-oriented filesystem path navigation

import pytest  # Testing framework for capturing exceptions and asserting conditions

from common.config import ConfigLoader  # Class/service responsible for loading and parsing YAML configs
from common.config.exceptions import (
    ConfigFileNotFoundError,        # Exception raised when the specified YAML file does not exist
    ConfigurationParsingError,      # Exception raised when YAML syntax is invalid
    InvalidConfigurationError,      # Exception raised when YAML schema or types are invalid
)
from common.config.types import ProjectConfig  # Root dataclass schema for overall project settings

# Path to the directory containing mock YAML files for testing
TEST_DATA_DIR = Path(__file__).parent / "test_data"


def test_load_valid_configuration() -> None:
    """
    Verify that a valid YAML configuration file is loaded into a strongly-typed ProjectConfig.
    """

    # Load and parse a known valid YAML configuration file
    config = ConfigLoader.load(TEST_DATA_DIR / "valid_config.yaml")

    # Verify dataset configuration parameters
    assert config.dataset.path == Path("data/raw")
    assert config.dataset.num_classes == 5

    # Verify training hyperparameter settings
    assert config.training.batch_size == 32

    # Verify neural network model selection
    assert config.model.architecture == "resnet50"

    # Verify global reproducible seed
    assert config.experiment.seed == 42


def test_missing_configuration_file() -> None:
    """
    Verify that attempting to load a non-existent configuration file raises ConfigFileNotFoundError.
    """

    # Attempting to load a non-existent file path must trigger ConfigFileNotFoundError
    with pytest.raises(ConfigFileNotFoundError):
        ConfigLoader.load(TEST_DATA_DIR / "does_not_exist.yaml")


def test_malformed_yaml() -> None:
    """
    Verify that loading a file with invalid YAML syntax raises ConfigurationParsingError.
    """

    # Passing a file with invalid YAML syntax must trigger ConfigurationParsingError
    with pytest.raises(ConfigurationParsingError):
        ConfigLoader.load(TEST_DATA_DIR / "malformed.yaml")


def test_invalid_root_object() -> None:
    """
    Verify that configuration files where the root object is not a mapping raise InvalidConfigurationError.
    """

    # YAML files whose top-level element is not a mapping/dict must trigger InvalidConfigurationError
    with pytest.raises(InvalidConfigurationError):
        ConfigLoader.load(TEST_DATA_DIR / "invalid_root.yaml")


def test_missing_required_section() -> None:
    """
    Verify that a configuration file missing a mandatory section raises InvalidConfigurationError.
    """

    # YAML files missing mandatory configuration sections must fail validation
    with pytest.raises(InvalidConfigurationError):
        ConfigLoader.load(TEST_DATA_DIR / "missing_section.yaml")