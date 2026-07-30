"""
Training entry point.

This module assembles the complete training pipeline by connecting all
project components:

- configuration loading
- dataset creation
- dataloader initialization
- model building
- loss function setup
- optimizer & scheduler construction
- metric building
- checkpoint management
- trainer execution
"""

from __future__ import annotations

import os
import sys

# Ensure project 'src' directory is on sys.path and remove script directory from sys.path
# BEFORE any standard library imports (like pathlib or argparse) to avoid shadowing 'types'.
_script_dir = os.path.dirname(os.path.abspath(__file__))
_src_dir = os.path.dirname(_script_dir)

if sys.path and os.path.abspath(sys.path[0]) == _script_dir:
    sys.path.pop(0)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

import argparse
import random
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from checkpoint.manager import CheckpointManager
from common.config.loader import ConfigLoader
from common.config.types import ProjectConfig
from dataloader.factory import (
    build_test_dataloader,
    build_train_dataloader,
    build_validation_dataloader,
)
from dataset.dataset import RetinalDataset
from metrics.metric_names import MetricName
from metrics.registry import get_metric
from trainer.trainer import Trainer
from training.loss_factory import LossFactory
from training.model_factory import ModelFactory
from training.optimizer_factory import OptimizerFactory
from training.scheduler_factory import SchedulerFactory

# ==========================================================
# Configuration & Paths
# ==========================================================

# Project root directory (3 levels up from src/trainer/train.py)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


# ==========================================================
# Reproducibility
# ==========================================================


def set_seed(seed: int) -> None:
    """
    Configure deterministic random seeds across all packages.

    Parameters
    ----------
    seed : int
        Global random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


# ==========================================================
# Configuration loading
# ==========================================================


def load_config(config_path: Path) -> ProjectConfig:
    """
    Load and parse project configuration.

    Parameters
    ----------
    config_path : Path
        YAML configuration file path.

    Returns
    -------
    ProjectConfig
        Parsed project configuration object.
    """
    resolved_path = (
        config_path
        if config_path.is_absolute()
        else (PROJECT_ROOT / config_path).resolve()
    )
    return ConfigLoader.load(resolved_path)


# ==========================================================
# Device selection
# ==========================================================


def resolve_device(requested_device: str) -> torch.device:
    """
    Resolve target execution computing device.

    Parameters
    ----------
    requested_device : str
        Device specified in configuration ("auto", "cuda", or "cpu").

    Returns
    -------
    torch.device
        Target PyTorch compute device.
    """
    device_str = requested_device.lower()
    if device_str == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA execution was requested in config, but PyTorch is installed without CUDA support "
                f"(torch.__version__ == {torch.__version__!r}). "
                "Please install PyTorch CUDA binaries."
            )
        return torch.device("cuda")
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


# ==========================================================
# Dataset & DataLoader Builders
# ==========================================================


def build_datasets(
    config: ProjectConfig,
) -> tuple[RetinalDataset, RetinalDataset, RetinalDataset]:
    """
    Build train, validation, and test RetinalDataset instances.

    Parameters
    ----------
    config : ProjectConfig
        Project configuration.

    Returns
    -------
    tuple[RetinalDataset, RetinalDataset, RetinalDataset]
        Train, validation, and test datasets.
    """
    dataset_root = (
        config.dataset.path
        if config.dataset.path.is_absolute()
        else (PROJECT_ROOT / config.dataset.path).resolve()
    )

    # Resolution order for split annotation files
    train_file = dataset_root / "train.txt"
    if not train_file.exists():
        train_file = dataset_root / config.dataset.annotation_file

    valid_file = dataset_root / "valid.txt"
    if not valid_file.exists():
        valid_file = train_file

    test_file = dataset_root / "test.txt"
    if not test_file.exists():
        test_file = train_file

    # Build dataclass overrides for split directories if they exist
    valid_img_dir = (
        "valid" if (dataset_root / "valid").is_dir() else config.dataset.image_directory
    )
    test_img_dir = (
        "test" if (dataset_root / "test").is_dir() else config.dataset.image_directory
    )

    valid_config = replace(config.dataset, image_directory=valid_img_dir)
    test_config = replace(config.dataset, image_directory=test_img_dir)

    # Build typed dataset instances
    train_dataset = RetinalDataset(
        config=config.dataset, annotation_file=train_file.name
    )
    valid_dataset = RetinalDataset(config=valid_config, annotation_file=valid_file.name)
    test_dataset = RetinalDataset(config=test_config, annotation_file=test_file.name)

    return train_dataset, valid_dataset, test_dataset


def build_dataloaders(
    config: ProjectConfig,
    train_dataset: RetinalDataset,
    validation_dataset: RetinalDataset,
    test_dataset: RetinalDataset,
):
    """
    Build DataLoaders for train, validation, and test splits.

    Parameters
    ----------
    config : ProjectConfig
        Project configuration.
    train_dataset : RetinalDataset
        Training dataset.
    validation_dataset : RetinalDataset
        Validation dataset.
    test_dataset : RetinalDataset
        Test dataset.

    Returns
    -------
    tuple
        Train, validation, and test DataLoaders.
    """
    train_loader = build_train_dataloader(
        dataset=train_dataset,
        config=config.dataloader,
    )
    validation_loader = build_validation_dataloader(
        dataset=validation_dataset,
        config=config.dataloader,
    )
    test_loader = build_test_dataloader(
        dataset=test_dataset,
        config=config.dataloader,
    )
    return train_loader, validation_loader, test_loader


# ==========================================================
# Metric Builder
# ==========================================================

_METRIC_NAME_MAP: dict[str, MetricName] = {
    "accuracy": MetricName.ACCURACY,
    "precision": MetricName.PRECISION,
    "recall": MetricName.RECALL,
    "f1": MetricName.F1,
    "macro_f1": MetricName.F1,
    "weighted_f1": MetricName.F1,
    "qwk": MetricName.QUADRATIC_WEIGHTED_KAPPA,
    "kappa": MetricName.QUADRATIC_WEIGHTED_KAPPA,
    "quadratic_weighted_kappa": MetricName.QUADRATIC_WEIGHTED_KAPPA,
    "confusion_matrix": MetricName.CONFUSION_MATRIX,
}


def build_metrics_dict(config: ProjectConfig) -> dict[str, Callable]:
    """
    Build dictionary of evaluation metric functions from config metrics list.

    Parameters
    ----------
    config : ProjectConfig
        Project configuration.

    Returns
    -------
    dict[str, Callable]
        Dictionary mapping metric names to evaluation functions.
    """
    metrics_dict: dict[str, Callable] = {}
    metric_keys = list(config.metrics.primary) + list(config.metrics.secondary)

    for key in metric_keys:
        clean_key = str(key).lower()
        if clean_key in _METRIC_NAME_MAP:
            enum_name = _METRIC_NAME_MAP[clean_key]
            metrics_dict[clean_key] = get_metric(enum_name)

    # Ensure accuracy is always available as fallback metric
    if "accuracy" not in metrics_dict:
        metrics_dict["accuracy"] = get_metric(MetricName.ACCURACY)

    return metrics_dict


# ==========================================================
# Training Components Builder
# ==========================================================


def build_training_components(config: ProjectConfig):
    """
    Construct model, criterion, optimizer, and scheduler.

    Parameters
    ----------
    config : ProjectConfig
        Project configuration.

    Returns
    -------
    tuple
        Model, criterion, optimizer, and scheduler instances.
    """
    model = ModelFactory.build(
        architecture=config.model.architecture,
        pretrained=config.model.pretrained,
        num_classes=config.model.num_classes,
        dropout_rate=getattr(config.model, "dropout_rate", 0.0),
    )

    class_weights = None
    if config.loss.name in ("class_balanced_focal", "weighted_cross_entropy"):
        from training.class_weights import compute_class_weights

        class_weights = compute_class_weights()

    criterion = LossFactory.build(
        name=config.loss.name,
        weight=class_weights,
        label_smoothing=getattr(config.loss, "label_smoothing", 0.0),
    )

    optimizer = OptimizerFactory.build(
        name=config.optimizer.name,
        parameters=model.parameters(),
        learning_rate=config.optimizer.lr,
        weight_decay=config.optimizer.weight_decay,
    )

    scheduler = SchedulerFactory.build(
        name=config.scheduler.name,
        optimizer=optimizer,
        epochs=config.training.epochs,
    )

    return model, criterion, optimizer, scheduler


# ==========================================================
# Trainer Builder
# ==========================================================


def build_trainer(
    config: ProjectConfig,
    train_loader,
    validation_loader,
    model,
    criterion,
    optimizer,
    scheduler,
    device: torch.device,
) -> Trainer:
    """
    Assemble the Trainer instance.

    Parameters
    ----------
    config : ProjectConfig
        Project configuration.
    train_loader
        Training DataLoader.
    validation_loader
        Validation DataLoader.
    model
        Neural network model.
    criterion
        Loss function.
    optimizer
        Optimizer.
    scheduler
        Learning rate scheduler.
    device : torch.device
        Compute device target.

    Returns
    -------
    Trainer
        Configured Trainer instance.
    """
    checkpoint_dir = (
        config.checkpoint.directory
        if config.checkpoint.directory.is_absolute()
        else (PROJECT_ROOT / config.checkpoint.directory).resolve()
    )

    checkpoint_manager = CheckpointManager(checkpoint_directory=checkpoint_dir)
    metrics_dict = build_metrics_dict(config)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        train_loader=train_loader,
        validation_loader=validation_loader,
        metrics=metrics_dict,
        checkpoint_manager=checkpoint_manager,
        device=device,
        mixed_precision=getattr(config.mixed_precision, "enabled", True),
        gradient_accumulation_steps=getattr(
            config.training, "gradient_accumulation_steps", 1
        ),
        validation_frequency=getattr(config.training, "validation_frequency", 1),
        early_stopping_patience=getattr(config.early_stopping, "patience", 10),
        early_stopping_min_delta=getattr(config.early_stopping, "min_delta", 0.0),
        log_every_n_steps=getattr(config.logging, "log_every_n_steps", 20),
    )

    return trainer


# ==========================================================
# Main CLI Entry Point
# ==========================================================


def main() -> None:
    """
    Project training CLI entry point.
    """
    parser = argparse.ArgumentParser(
        description="Retinal Disease Detection - Model Fine-tuning Pipeline"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to YAML configuration file",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("RETINAL DISEASE DETECTION - TRAINING PIPELINE")
    print("=" * 60)

    # 1. Configuration & Reproducibility
    config = load_config(args.config)
    set_seed(config.experiment.seed)
    device = resolve_device(config.training.device)

    print(f"Config File : {args.config}")
    print(f"Architecture: {config.model.architecture}")
    print(f"Target Device: {device}")
    print(f"Random Seed : {config.experiment.seed}")

    # 2. Dataset & DataLoaders
    train_ds, val_ds, test_ds = build_datasets(config)
    print(
        f"\nDataset Samples -> Train: {len(train_ds)} | Valid: {len(val_ds)} | Test: {len(test_ds)}"
    )

    train_loader, val_loader, test_loader = build_dataloaders(
        config, train_ds, val_ds, test_ds
    )

    # 3. Model, Loss, Optimizer, Scheduler
    model, criterion, optimizer, scheduler = build_training_components(config)

    # 4. Trainer
    trainer = build_trainer(
        config=config,
        train_loader=train_loader,
        validation_loader=val_loader,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
    )

    # 5. Execute Fine-Tuning
    print("\nStarting Model Fine-Tuning...")
    monitor_metric = getattr(config.checkpoint, "monitor", "val_loss")
    training_summary = trainer.fit(
        epochs=config.training.epochs,
        monitor=monitor_metric,
    )

    print("\nFine-Tuning Finished!")
    print(f"Best Epoch : {training_summary.best_epoch}")
    print(f"Best Score : {training_summary.best_metric:.4f}")

    # 6. Evaluate Test Set
    print("\nEvaluating on Test Set...")
    test_epoch_out = trainer.test(test_loader)
    test_metrics = {"test_loss": float(test_epoch_out.loss), **test_epoch_out.metrics}

    # 7. Generate Visual Report
    print("\nGenerating Automated Visual Report...")
    from visualization.report_generator import HTMLReportGenerator

    report_file = (
        PROJECT_ROOT
        / "reports"
        / "metrics"
        / f"experiment_{config.experiment.name}_report.html"
    )

    config_summary = {
        "Architecture": config.model.architecture,
        "Optimizer": config.optimizer.name,
        "Learning Rate": str(config.optimizer.lr),
        "Loss Function": config.loss.name,
        "Batch Size": str(config.dataloader.batch_size),
        "Total Epochs": str(config.training.epochs),
    }

    # Extract history if available from trainer.state
    history = {}
    if hasattr(trainer, "state"):
        history["train_loss"] = trainer.state.train_loss_history
        history["val_loss"] = trainer.state.val_loss_history
        if "val_qwk" in trainer.state.val_metrics_history:
            history["val_qwk"] = trainer.state.val_metrics_history["val_qwk"]
        if "val_accuracy" in trainer.state.val_metrics_history:
            history["val_accuracy"] = trainer.state.val_metrics_history["val_accuracy"]

    HTMLReportGenerator.generate(
        experiment_name=config.experiment.name,
        metrics=test_metrics,
        config_summary=config_summary,
        history=history,
        output_path=report_file,
    )
    print(f"Visual HTML Report exported successfully to: {report_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
