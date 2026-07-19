"""Unit tests for the configuration loader."""

from pathlib import Path

import pytest

from common.config import ConfigLoader
from common.config.exceptions import (
    ConfigFileNotFoundError,
    ConfigurationParsingError,
    InvalidConfigurationError,
)
from common.config.types import ProjectConfig

TEST_DATA_DIR = Path(__file__).parent / "test_data"


def test_load_valid_configuration() -> None:
    """Verify that a valid configuration is loaded successfully."""

    config = ConfigLoader.load(TEST_DATA_DIR / "valid_config.yaml")

    assert config.dataset.path == Path("data/raw")
    assert config.dataset.num_classes == 5
    assert config.training.batch_size == 32
    assert config.model.architecture == "resnet50"
    assert config.experiment.seed == 42


def test_missing_configuration_file() -> None:
    """Verify that loading a missing configuration file raises an exception."""

    with pytest.raises(ConfigFileNotFoundError):
        ConfigLoader.load(TEST_DATA_DIR / "does_not_exist.yaml")


def test_malformed_yaml() -> None:
    """Verify that malformed YAML raises a parsing exception."""

    with pytest.raises(ConfigurationParsingError):
        ConfigLoader.load(TEST_DATA_DIR / "malformed.yaml")


def test_invalid_root_object() -> None:
    """Verify that the configuration root must be a dictionary."""

    with pytest.raises(InvalidConfigurationError):
        ConfigLoader.load(TEST_DATA_DIR / "invalid_root.yaml")


def test_missing_required_section() -> None:
    """Verify that a missing section raises an exception."""

    with pytest.raises(InvalidConfigurationError):
        ConfigLoader.load(TEST_DATA_DIR / "missing_section.yaml")
