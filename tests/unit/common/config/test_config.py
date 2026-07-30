"""Unit tests for the configuration sub-system (types, loader, exceptions, validator)."""

from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from src.common.config.exceptions import (
    ConfigFileNotFoundError,
    ConfigurationParsingError,
    InvalidConfigurationError,
)
from src.common.config.loader import ConfigLoader
from src.common.config.types import ProjectConfig
from src.common.config.validator import validate_config


@pytest.fixture
def valid_config_dict(tmp_path: Path) -> dict:
    """Fixture providing a complete and valid configuration dictionary."""
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    (dataset_dir / "train.csv").touch()

    return {
        "dataset": {
            "path": str(dataset_dir),
            "annotation_file": "train.csv",
            "image_directory": "images",
            "num_classes": 3,
        },
        "preprocessing": {
            "image_size": [224, 224],
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "horizontal_flip_prob": 0.5,
            "vertical_flip_prob": 0.0,
            "rotation_limit": 15,
            "rotation_prob": 0.3,
            "brightness_contrast_prob": 0.2,
        },
        "dataloader": {
            "batch_size": 32,
            "shuffle": True,
            "num_workers": 4,
            "pin_memory": True,
            "drop_last": False,
            "persistent_workers": True,
            "weight_class_balance": False,
        },
        "training": {
            "epochs": 10,
            "device": "cuda",
        },
        "model": {
            "architecture": "resnet18",
            "pretrained": True,
            "num_classes": 3,
        },
        "loss": {
            "name": "focal",
            "class_weights": False,
            "gamma": 2.0,
            "alpha": None,
            "beta": 0.9999,
            "reduction": "mean",
        },
        "metrics": {
            "primary": ["qwk"],
            "secondary": ["macro_f1", "accuracy"],
            "per_class": True,
        },
        "experiment": {
            "name": "baseline_run",
            "seed": 42,
        },
    }


@pytest.fixture
def valid_yaml_file(tmp_path: Path, valid_config_dict: dict) -> Path:
    """Fixture to generate a temporary valid YAML config file."""
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(valid_config_dict, f)
    return config_path


# =====================================================================
# 1. Tests for ConfigLoader & File Operations
# =====================================================================


def test_load_valid_config(valid_yaml_file: Path) -> None:
    """Test successful loading of a valid YAML configuration."""
    config = ConfigLoader.load(valid_yaml_file)

    assert isinstance(config, ProjectConfig)
    assert config.dataset.num_classes == 3
    assert config.preprocessing.image_size == (224, 224)
    assert config.dataloader.batch_size == 32
    assert config.training.epochs == 10
    assert config.model.architecture == "resnet18"


def test_load_missing_file(tmp_path: Path) -> None:
    """Test that ConfigFileNotFoundError is raised with proper attributes when file does not exist."""
    non_existent_file = tmp_path / "missing.yaml"

    with pytest.raises(ConfigFileNotFoundError) as exc_info:
        ConfigLoader.load(non_existent_file)

    assert exc_info.value.file_path == non_existent_file
    assert "Configuration file not found" in str(exc_info.value)


def test_load_malformed_yaml(tmp_path: Path) -> None:
    """Test that ConfigurationParsingError is raised for syntax errors in YAML."""
    invalid_yaml_path = tmp_path / "invalid.yaml"
    invalid_yaml_path.write_text("dataset: [unclosed_list", encoding="utf-8")

    with pytest.raises(ConfigurationParsingError) as exc_info:
        ConfigLoader.load(invalid_yaml_path)

    assert exc_info.value.file_path == invalid_yaml_path


def test_load_empty_yaml(tmp_path: Path) -> None:
    """Test handling of completely empty YAML files."""
    empty_yaml_path = tmp_path / "empty.yaml"
    empty_yaml_path.write_text("", encoding="utf-8")

    with pytest.raises(InvalidConfigurationError) as exc_info:
        ConfigLoader.load(empty_yaml_path)

    assert exc_info.value.param_name == "root"
    assert "is empty" in exc_info.value.reason


def test_image_size_scalar_conversion(tmp_path: Path, valid_config_dict: dict) -> None:
    """Test that a scalar integer image_size is converted to an (int, int) tuple."""
    valid_config_dict["preprocessing"]["image_size"] = 512
    config_path = tmp_path / "scalar_size.yaml"

    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(valid_config_dict, f)

    config = ConfigLoader.load(config_path)
    assert config.preprocessing.image_size == (512, 512)


# =====================================================================
# 2. Tests for Validator Boundaries & Business Logic
# =====================================================================


def test_validate_non_existent_dataset_path(valid_yaml_file: Path) -> None:
    """Test validation failure when dataset directory path doesn't exist."""
    config = ConfigLoader.load(valid_yaml_file)

    # Создаем модифицированную копию замороженного объекта
    invalid_dataset = replace(
        config.dataset, path=Path("/non/existent/path/for/dataset")
    )
    invalid_config = replace(config, dataset=invalid_dataset)

    with pytest.raises(InvalidConfigurationError) as exc_info:
        validate_config(invalid_config)

    assert exc_info.value.param_name == "dataset.path"


def test_validate_mismatched_num_classes(valid_yaml_file: Path) -> None:
    """Test validation failure when model num_classes doesn't match dataset num_classes."""
    config = ConfigLoader.load(valid_yaml_file)

    invalid_model = replace(config.model, num_classes=10)
    invalid_config = replace(config, model=invalid_model)

    with pytest.raises(InvalidConfigurationError) as exc_info:
        validate_config(invalid_config)

    assert exc_info.value.param_name == "model.num_classes"


def test_validate_invalid_probability_range(valid_yaml_file: Path) -> None:
    """Test validation failure for out-of-bound probabilities (> 1.0)."""
    config = ConfigLoader.load(valid_yaml_file)

    invalid_prep = replace(config.preprocessing, horizontal_flip_prob=1.5)
    invalid_config = replace(config, preprocessing=invalid_prep)

    with pytest.raises(InvalidConfigurationError) as exc_info:
        validate_config(invalid_config)

    assert exc_info.value.param_name == "preprocessing.horizontal_flip_prob"


def test_validate_unsupported_loss_name(valid_yaml_file: Path) -> None:
    """Test validation failure when an unknown loss function is specified."""
    config = ConfigLoader.load(valid_yaml_file)

    invalid_loss = replace(config.loss, name="unsupported_magic_loss")
    invalid_config = replace(config, loss=invalid_loss)

    with pytest.raises(InvalidConfigurationError) as exc_info:
        validate_config(invalid_config)

    assert exc_info.value.param_name == "loss.name"


def test_validate_invalid_dataloader_batch_size(valid_yaml_file: Path) -> None:
    """Test validation failure for non-positive DataLoader batch_size."""
    config = ConfigLoader.load(valid_yaml_file)

    invalid_dataloader = replace(config.dataloader, batch_size=0)
    invalid_config = replace(config, dataloader=invalid_dataloader)

    with pytest.raises(InvalidConfigurationError) as exc_info:
        validate_config(invalid_config)

    assert exc_info.value.param_name == "dataloader.batch_size"
