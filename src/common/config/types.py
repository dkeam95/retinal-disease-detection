"""Type definitions for the configuration system.

This module contains immutable configuration objects used
throughout the project.
"""
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
    """Preprocessing configuration."""

    pipeline: list[str]


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Training configuration."""

    batch_size: int
    epochs: int
    num_workers: int
    device: str


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Model configuration"""

    name: str
    pretrained: bool


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
