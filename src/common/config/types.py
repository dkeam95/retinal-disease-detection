"""Type definitions for the configuration system.

This module contains immutable configuration objects used
throughout the project.
"""
from typing import Literal
from pathlib import Path
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    """Dataset configuration.
    
    Attributes:
    path: Root directory of the dataset.
    annotation_file: Annotation filename (e.g. "train.txt").
    image_directory: Directory containing dataset images.
    num_classes:Number of dataset classes.
    """

    path: Path
    annotation_file: str
    image_directory: str
    num_classes: int


@dataclass(frozen=True, slots=True)
class PreprocessingConfig:
    """Image preprocessing configuration.

    Attributes:
        image_size:
            Target image size.

        mean:
            Channel-wise mean used for normalization.

        std:
            Channel-wise standard deviation used for normalization.

        horizontal_flip_prob:
            Probability of applying horizontal flip.

        vertical_flip_prob:
            Probability of applying vertical flip.

        rotation_limit:
            Maximum rotation angle in degrees.

        brightness_contrast_prob:
            Probability of applying brightness/contrast augmentation.
    """

    image_size: int

    mean: tuple[float, float, float]
    std: tuple[float, float, float]

    horizontal_flip_prob: float
    vertical_flip_prob: float

    rotation_limit: int
    rotation_prob: float

    brightness_contrast_prob: float



@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Training configuration."""

    batch_size: int
    epochs: int
    num_workers: int
    device: Literal[
        "cpu",
        "cuda",
        "mps"
    ]


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Model configuration.

    Attributes:
        name: Model architecture name (e.g. "resnet18", "densenet121").
        pretrained: Whether to use pre-trained weights.
        num_classes: Number of output classes.
    """

    architecture: str
    pretrained: bool
    num_classes: int


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Experiment configuration."""

    name: str
    seed: int


@dataclass(frozen=True, slots=True)
class ProjectConfig:
    """Root project configuration."""

    dataset: DatasetConfig
    preprocessing: PreprocessingConfig
    training: TrainingConfig
    model: ModelConfig
    experiment: ExperimentConfig
