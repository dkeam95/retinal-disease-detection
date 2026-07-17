"""Type definitions for the configuration system.

This module contains immutable configuration objects used
throughout the project.
"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    """Dataset configuration."""

    path: str
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
