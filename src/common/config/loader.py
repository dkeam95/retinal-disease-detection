"""Configuration loader for the project.

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
    CheckpointConfig,
    DataLoaderConfig,
    DatasetConfig,
    DetectionDatasetConfig,
    DetectionModelConfig,
    EarlyStoppingConfig,
    ExperimentConfig,
    LoggingConfig,
    LossConfig,
    MetricsConfig,
    MixedPrecisionConfig,
    ModelConfig,
    OptimizerConfig,
    PreprocessingConfig,
    ProjectConfig,
    SchedulerConfig,
    TrainingConfig,
)
from .validator import validate_config


class ConfigLoader:
    """Loads and parses project configuration files."""

    @staticmethod
    def load(path: str | Path) -> ProjectConfig:
        """Load and validate a project configuration.

        Args:
            path: Path to the YAML configuration file.

        Returns:
            Validated ProjectConfig object.

        Raises:
            ConfigFileNotFoundError: If the file does not exist on disk.
            ConfigurationParsingError: If YAML contains syntax errors.
            InvalidConfigurationError: If missing keys or invalid structures exist.
        """
        data = ConfigLoader._read_yaml(path)
        config = ConfigLoader._to_project_config(data, file_path=Path(path))
        validate_config(config)
        return config

    @staticmethod
    def _read_yaml(path: str | Path) -> dict[str, Any]:
        """Read a YAML configuration file safely from disk.

        Args:
            path: Target file path.

        Returns:
            Dictionary containing parsed YAML key-value pairs.
        """
        file_path = Path(path)

        if not file_path.exists():
            raise ConfigFileNotFoundError(file_path)

        try:
            with file_path.open("r", encoding="utf-8") as file:
                data = yaml.safe_load(file)
        except yaml.YAMLError as error:
            raise ConfigurationParsingError(file_path, str(error)) from error

        if data is None:
            raise InvalidConfigurationError(
                param_name="root",
                value=None,
                reason=f"Configuration file '{file_path.name}' is empty.",
            )

        if not isinstance(data, dict):
            raise InvalidConfigurationError(
                param_name="root",
                value=type(data).__name__,
                reason=f"Configuration root in '{file_path.name}' must be a dictionary.",
            )

        return data

    @staticmethod
    def _parse_image_size(value: Any) -> tuple[int, int]:
        """Convert scalar or list image size value to a (height, width) tuple."""
        if isinstance(value, int):
            return (value, value)
        if isinstance(value, (list, tuple)) and len(value) == 2:
            return (int(value[0]), int(value[1]))
        raise ValueError(
            f"Expected int or sequence of 2 integers for image_size, got {value}"
        )

    @staticmethod
    def _to_project_config(data: dict[str, Any], file_path: Path) -> ProjectConfig:
        """Convert a raw dictionary into a strongly typed ProjectConfig object."""
        try:
            prep_data = data["preprocessing"]
            detection_dataset_data = data.get("detection_dataset")
            detection_model_data = data.get("detection_model")

            # Parse image size flexibly from image_size or resize dict
            if "image_size" in prep_data:
                image_size = ConfigLoader._parse_image_size(prep_data["image_size"])
            elif "resize" in prep_data and isinstance(prep_data["resize"], dict):
                image_size = (
                    int(prep_data["resize"]["height"]),
                    int(prep_data["resize"]["width"]),
                )
            else:
                image_size = (224, 224)

            # Parse normalization parameters flexibly
            if "normalize" in prep_data and isinstance(prep_data["normalize"], dict):
                mean = tuple(prep_data["normalize"]["mean"])
                std = tuple(prep_data["normalize"]["std"])
            else:
                mean = tuple(prep_data.get("mean", [0.485, 0.456, 0.406]))
                std = tuple(prep_data.get("std", [0.229, 0.224, 0.225]))

            return ProjectConfig(
                dataset=DatasetConfig(
                    path=Path(data["dataset"]["path"]),
                    annotation_file=str(data["dataset"]["annotation_file"]),
                    image_directory=str(data["dataset"]["image_directory"]),
                    num_classes=int(data["dataset"]["num_classes"]),
                ),
                preprocessing=PreprocessingConfig(
                    image_size=image_size,
                    mean=mean,
                    std=std,
                    horizontal_flip_prob=float(
                        prep_data.get("horizontal_flip_prob", 0.5)
                    ),
                    vertical_flip_prob=float(prep_data.get("vertical_flip_prob", 0.0)),
                    rotation_limit=int(prep_data.get("rotation_limit", 15)),
                    rotation_prob=float(prep_data.get("rotation_prob", 0.3)),
                    brightness_contrast_prob=float(
                        prep_data.get("brightness_contrast_prob", 0.2)
                    ),
                ),
                dataloader=DataLoaderConfig(
                    batch_size=int(data["dataloader"]["batch_size"]),
                    shuffle=bool(data["dataloader"]["shuffle"]),
                    num_workers=int(data["dataloader"]["num_workers"]),
                    pin_memory=bool(data["dataloader"]["pin_memory"]),
                    drop_last=bool(data["dataloader"]["drop_last"]),
                    persistent_workers=bool(data["dataloader"]["persistent_workers"]),
                    weight_class_balance=bool(
                        data["dataloader"].get("weight_class_balance", False)
                    ),
                ),
                training=TrainingConfig(
                    epochs=int(data["training"]["epochs"]),
                    device=data["training"]["device"],
                ),
                model=ModelConfig(
                    architecture=str(data["model"]["architecture"]),
                    pretrained=bool(data["model"]["pretrained"]),
                    num_classes=int(data["model"]["num_classes"]),
                    dropout_rate=float(data["model"].get("dropout_rate", 0.0)),
                ),
                loss=LossConfig(
                    name=str(data["loss"]["name"]),
                    class_weights=bool(data["loss"].get("class_weights", False)),
                    gamma=float(data["loss"].get("gamma", 2.0)),
                    alpha=data["loss"].get("alpha"),
                    beta=float(data["loss"].get("beta", 0.9999)),
                    reduction=str(data["loss"].get("reduction", "mean")),
                ),
                metrics=MetricsConfig(
                    primary=tuple(
                        data.get("metrics", {}).get("primary", ["accuracy", "qwk"])
                    ),
                    secondary=tuple(
                        data.get("metrics", {}).get("secondary", ["macro_f1"])
                    ),
                    per_class=bool(data.get("metrics", {}).get("per_class", True)),
                ),
                experiment=ExperimentConfig(
                    name=str(data["experiment"]["name"]),
                    seed=int(data["experiment"]["seed"]),
                ),
                optimizer=OptimizerConfig(
                    name=str(data.get("optimizer", {}).get("name", "adamw")),
                    lr=float(
                        data.get("optimizer", {}).get(
                            "learning_rate", data.get("optimizer", {}).get("lr", 0.0003)
                        )
                    ),
                    weight_decay=float(
                        data.get("optimizer", {}).get("weight_decay", 0.0001)
                    ),
                ),
                scheduler=SchedulerConfig(
                    name=str(data.get("scheduler", {}).get("name", "cosine")),
                    eta_min=float(data.get("scheduler", {}).get("eta_min", 1.0e-6)),
                ),
                checkpoint=CheckpointConfig(
                    directory=Path(
                        data.get("checkpoint", {}).get("directory", "checkpoints")
                    ),
                    monitor=str(data.get("checkpoint", {}).get("monitor", "val_loss")),
                ),
                early_stopping=EarlyStoppingConfig(
                    enabled=bool(data.get("early_stopping", {}).get("enabled", True)),
                    patience=int(data.get("early_stopping", {}).get("patience", 10)),
                    min_delta=float(
                        data.get("early_stopping", {}).get("min_delta", 0.001)
                    ),
                ),
                mixed_precision=MixedPrecisionConfig(
                    enabled=bool(data.get("mixed_precision", {}).get("enabled", True)),
                    dtype=str(data.get("mixed_precision", {}).get("dtype", "float16")),
                ),
                logging=LoggingConfig(
                    log_every_n_steps=int(
                        data.get("logging", {}).get("log_every_n_steps", 20)
                    ),
                ),
                detection_dataset=(
                    DetectionDatasetConfig(
                        xml_dir=Path(detection_dataset_data["xml_dir"]),
                        image_dir=Path(detection_dataset_data["image_dir"]),
                        num_classes=int(detection_dataset_data.get("num_classes", 5)),
                    )
                    if detection_dataset_data is not None
                    else None
                ),
                detection_model=(
                    DetectionModelConfig(
                        architecture=str(
                            detection_model_data.get(
                                "architecture", "fasterrcnn_resnet50_fpn"
                            )
                        ),
                        pretrained=bool(detection_model_data.get("pretrained", True)),
                        num_classes=int(detection_model_data.get("num_classes", 5)),
                        score_thresh=float(detection_model_data.get("score_thresh", 0.25)),
                        nms_thresh=float(detection_model_data.get("nms_thresh", 0.45)),
                        min_size=int(detection_model_data.get("min_size", 640)),
                        max_size=int(detection_model_data.get("max_size", 640)),
                    )
                    if detection_model_data is not None
                    else None
                ),
            )

        except KeyError as error:
            missing_key = error.args[0]
            raise InvalidConfigurationError(
                param_name=str(missing_key),
                value="Missing",
                reason=f"Missing required configuration key in '{file_path.name}'.",
            ) from error
        except (TypeError, ValueError) as error:
            raise InvalidConfigurationError(
                param_name="structure",
                value="Malformed",
                reason=f"Type conversion or formatting failed in '{file_path.name}': {error}",
            ) from error
