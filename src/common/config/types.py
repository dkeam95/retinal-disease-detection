"""Type definitions for the configuration system.

This module contains immutable configuration objects used
throughout the project.
"""

from dataclasses import dataclass  # Decorator to automatically generate special methods (__init__, __repr__, etc.)
from pathlib import Path           # Object-oriented filesystem paths
from typing import Literal         # Type hint for restricting string values to a specific set of allowed options


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    """Dataset configuration.
    
    Attributes:
        path:
           Root directory of the dataset.
        annotation_file:
           Annotation filename (e.g. "train.txt").
        image_directory:
           Directory containing dataset images.
        num_classes:
           Number of dataset classes.
    """

    path: Path            # Root directory path on disk
    annotation_file: str  # Name of annotation/label file
    image_directory: str  # Subfolder containing raw images
    num_classes: int      # Total number of target classification categories


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
        rotation_prob:
           Probability of applying rotation.
        brightness_contrast_prob:
           Probability of applying brightness/contrast augmentation.
    """

    image_size: int  # Spatial dimension (height/width) for image resizing

    mean: tuple[float, float, float]  # RGB normalization mean values
    std: tuple[float, float, float]   # RGB normalization standard deviation values

    horizontal_flip_prob: float       # Augmentation probability for horizontal flipping [0.0, 1.0]
    vertical_flip_prob: float         # Augmentation probability for vertical flipping [0.0, 1.0]

    rotation_limit: int               # Maximum angle threshold for random image rotations
    rotation_prob: float              # Augmentation probability for applying rotation [0.0, 1.0]

    brightness_contrast_prob: float   # Probability of adjusting brightness/contrast [0.0, 1.0]


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Training configuration.
    
    Attributes:
        batch_size:
            Number of samples per batch.
        epochs:
            Number of training epochs.
        num_workers:
            Number of data loader workers.
        device:
            Device to train on.
    """

    batch_size: int  # Number of training images per optimization step
    epochs: int      # Total passes through the full training dataset
    num_workers: int # Subprocess count for parallel PyTorch DataLoader fetching
    device: Literal[
        "cpu",
        "cuda",
        "mps"
    ]  # Allowed execution hardware targets (CPU, NVIDIA CUDA GPU, or Apple Silicon MPS)


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Model configuration.

    Attributes:
        architecture: 
            Model architecture name (e.g. "resnet18", "densenet121").
        pretrained: 
            Whether to use pre-trained weights.
        num_classes: 
            Number of output classes.
    """

    architecture: str  # Backbone model identifier string
    pretrained: bool   # Flag to initialize backbone with ImageNet pretrained weights
    num_classes: int   # Number of output neurons in the final classification layer


@dataclass(frozen=True, slots=True)
class LossConfig:
    """Loss function configuration.

    Attributes:
        name: 
            Loss function name.
        class_weights: 
            Whether to use class weights.
        gamma: 
            Gamma parameter for focal loss.
        alpha: 
            Alpha parameter for focal loss.
        beta: 
            Beta parameter for focal loss.
        reduction: 
            Reduction method for loss.
    """

    name: str                    # Loss function identifier (e.g., "ce", "focal", "cb_focal")
    class_weights: bool = False  # Flag to enable loss weighting by inverse class frequencies
    gamma: float = 2.0           # Focusing parameter modulating easy vs hard examples in Focal Loss
    alpha: float | None = None   # Optional balancing factor for class imbalance in Focal Loss
    beta: float = 0.9999         # Hyperparameter for Class-Balanced loss calculation
    reduction: str = "mean"      # Batch loss reduction technique ("mean", "sum", or "none")


@dataclass(frozen=True, slots=True)
class MetricsConfig:
    """Metrics configuration.
    
    Attributes:
        primary:
            Metrics used for model selection.

        secondary:
            Additional metrics.

        per_class:
            Whether to compute per-class metrics.
    """

    primary: tuple[str, ...]    # Tuple of main performance metrics for model checkpointing
    secondary: tuple[str, ...]  # Tuple of auxiliary metrics to track during evaluation
    per_class: bool = True      # Flag to compute breakdown metrics for each individual class


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Experiment configuration.

    Attributes:
        name: 
            Experiment name.
        seed: 
            Random seed for reproducibility.
    """

    name: str  # Unique experiment identifier for logging and tracking
    seed: int  # Global random seed ensuring deterministic training runs


@dataclass(frozen=True, slots=True)
class ProjectConfig:
    """Root project configuration.
    
    Attributes:
        dataset: 
            Dataset configuration.
        preprocessing: 
            Preprocessing configuration.
        training: 
            Training configuration.
        model: 
            Model configuration.
        loss: 
            Loss function configuration.
        metrics: 
            Metrics configuration.
        experiment: 
            Experiment configuration.
    """

    dataset: DatasetConfig            # Nested dataset configuration object
    preprocessing: PreprocessingConfig# Nested image preprocessing and augmentation settings
    training: TrainingConfig          # Nested training hyperparameters and device runtime settings
    model: ModelConfig                # Nested neural network architecture properties
    loss: LossConfig                  # Nested loss function hyperparameter configuration
    metrics: MetricsConfig            # Nested evaluation metrics settings
    experiment: ExperimentConfig      # Nested experiment logging and seeding properties