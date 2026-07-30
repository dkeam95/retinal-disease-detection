"""Type definitions for the configuration system.

This module contains immutable configuration objects used
throughout the project.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    """Dataset configuration.

    Attributes:
        path: Root directory of the dataset.
        annotation_file: Annotation filename (e.g. "train.txt").
        image_directory: Directory containing dataset images.
        num_classes: Number of dataset classes.
    """

    path: Path
    annotation_file: str
    image_directory: str
    num_classes: int


@dataclass(frozen=True, slots=True)
class PreprocessingConfig:
    """Image preprocessing configuration.

    Attributes:
        image_size: Target image size (height, width).
        mean: Channel-wise mean used for normalization.
        std: Channel-wise standard deviation used for normalization.
        horizontal_flip_prob: Probability of applying horizontal flip.
        vertical_flip_prob: Probability of applying vertical flip.
        rotation_limit: Maximum rotation angle in degrees.
        rotation_prob: Probability of applying rotation.
        brightness_contrast_prob: Probability of applying brightness/contrast augmentation.
    """

    image_size: tuple[int, int]
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    horizontal_flip_prob: float
    vertical_flip_prob: float
    rotation_limit: int
    rotation_prob: float
    brightness_contrast_prob: float


@dataclass(frozen=True, slots=True)
class DataLoaderConfig:
    """DataLoader configuration parameters.

    Attributes:
        batch_size: Number of training images per optimization step.
        shuffle: Whether to shuffle the data.
        num_workers: Subprocess count for parallel PyTorch DataLoader fetching.
        pin_memory: Whether to pin memory for faster GPU transfer.
        drop_last: Whether to drop the last incomplete batch.
        persistent_workers: Whether to keep worker processes alive between epochs.
        weight_class_balance: Whether to weight the classes.
    """

    batch_size: int
    shuffle: bool
    num_workers: int
    pin_memory: bool
    drop_last: bool
    persistent_workers: bool
    weight_class_balance: bool = False


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Training optimization and runtime configuration.

    Attributes:
        epochs: Total passes through the full training dataset.
        device: Allowed execution hardware targets (CPU, NVIDIA CUDA GPU, or Apple Silicon MPS).
    """

    epochs: int
    device: Literal["cpu", "cuda", "mps"]


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Neural network architecture configuration.

    Attributes:
        architecture: Backbone model identifier string.
        pretrained: Flag to initialize backbone with ImageNet pretrained weights.
        num_classes: Number of output neurons in the final classification layer.
    """

    architecture: str
    pretrained: bool
    num_classes: int
    dropout_rate: float = 0.0


@dataclass(frozen=True, slots=True)
class LossConfig:
    """Loss function configuration.

    Attributes:
        name: Loss function identifier (e.g., "ce", "focal", "cb_focal").
        class_weights: Flag to enable loss weighting by inverse class frequencies.
        gamma: Focusing parameter modulating easy vs hard examples in Focal Loss.
        alpha: Optional balancing factor for class imbalance in Focal Loss.
        beta: Hyperparameter for Class-Balanced loss calculation.
        reduction: Batch loss reduction technique ("mean", "sum", or "none").
    """

    name: str
    class_weights: bool = False
    gamma: float = 2.0
    alpha: float | None = None
    beta: float = 0.9999
    reduction: str = "mean"


@dataclass(frozen=True, slots=True)
class MetricsConfig:
    """Evaluation metrics configuration.

    Attributes:
        primary: Tuple of main performance metrics for model checkpointing.
        secondary: Tuple of auxiliary metrics to track during evaluation.
        per_class: Flag to compute breakdown metrics for each individual class.
    """

    primary: tuple[str, ...]
    secondary: tuple[str, ...]
    per_class: bool = True


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Experiment metadata and reproducibility configuration.

    Attributes:
        name: Unique experiment identifier for logging and tracking.
        seed: Global random seed ensuring deterministic training runs.
    """

    name: str
    seed: int


@dataclass(frozen=True, slots=True)
class LossConfig:
    """Loss function configuration.

    Attributes:
        name: Loss function identifier (e.g., "ce", "focal", "cb_focal").
        class_weights: Flag to enable loss weighting by inverse class frequencies.
        gamma: Focusing parameter modulating easy vs hard examples in Focal Loss.
        alpha: Optional balancing factor for class imbalance in Focal Loss.
        beta: Hyperparameter for Class-Balanced loss calculation.
        reduction: Batch loss reduction technique ("mean", "sum", or "none").
    """

    name: str
    class_weights: bool = False
    gamma: float = 2.0
    alpha: float | None = None
    beta: float = 0.9999
    reduction: str = "mean"


@dataclass(frozen=True, slots=True)
class MetricsConfig:
    """Evaluation metrics configuration.

    Attributes:
        primary: Tuple of main performance metrics for model checkpointing.
        secondary: Tuple of auxiliary metrics to track during evaluation.
        per_class: Flag to compute breakdown metrics for each individual class.
    """

    primary: tuple[str, ...]
    secondary: tuple[str, ...]
    per_class: bool = True


@dataclass(frozen=True, slots=True)
class OptimizerConfig:
    name: str = "adamw"
    lr: float = 0.0003
    weight_decay: float = 0.0001


@dataclass(frozen=True, slots=True)
class SchedulerConfig:
    name: str = "cosine"
    eta_min: float = 1.0e-6


@dataclass(frozen=True, slots=True)
class CheckpointConfig:
    directory: Path = Path("checkpoints")
    monitor: str = "val_loss"


@dataclass(frozen=True, slots=True)
class EarlyStoppingConfig:
    enabled: bool = True
    patience: int = 10
    min_delta: float = 0.001


@dataclass(frozen=True, slots=True)
class MixedPrecisionConfig:
    enabled: bool = True
    dtype: str = "float16"


@dataclass(frozen=True, slots=True)
class LoggingConfig:
    log_every_n_steps: int = 20


@dataclass(frozen=True, slots=True)
class ProjectConfig:
    """Root project configuration aggregating all subsystem configs.

    Attributes:
        dataset: Dataset configuration.
        preprocessing: Image preprocessing and augmentation settings.
        dataloader: DataLoader configuration.
        training: Training hyperparameters and device runtime settings.
        model: Neural network architecture properties.
        loss: Loss function hyperparameter configuration.
        metrics: Evaluation metrics settings.
        experiment: Experiment logging and seeding properties.
        optimizer: Optimizer hyperparameter configuration.
        scheduler: Learning rate scheduler configuration.
        checkpoint: Checkpoint management configuration.
        early_stopping: Early stopping runtime configuration.
        mixed_precision: Automatic mixed precision settings.
        logging: Logging configuration.
    """

    dataset: DatasetConfig
    preprocessing: PreprocessingConfig
    dataloader: DataLoaderConfig
    training: TrainingConfig
    model: ModelConfig
    loss: LossConfig
    metrics: MetricsConfig
    experiment: ExperimentConfig
    optimizer: OptimizerConfig = OptimizerConfig()
    scheduler: SchedulerConfig = SchedulerConfig()
    checkpoint: CheckpointConfig = CheckpointConfig()
    early_stopping: EarlyStoppingConfig = EarlyStoppingConfig()
    mixed_precision: MixedPrecisionConfig = MixedPrecisionConfig()
    logging: LoggingConfig = LoggingConfig()
