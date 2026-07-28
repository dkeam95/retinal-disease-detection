"""Type definitions for the training module.

This module defines lightweight dataclass containers used to group together
training components and data artifacts before the training process starts.
"""

from __future__ import annotations

from dataclasses import dataclass

from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader, Dataset


@dataclass(slots=True)
class TrainingArtifacts:
    """
    Container holding dataset-related training artifacts.

    Attributes
    ----------
    dataset : Dataset
        Full training dataset.

    train_loader : DataLoader
        DataLoader used for training.

    validation_loader : DataLoader
        DataLoader used for validation.
    """

    dataset: Dataset
    train_loader: DataLoader
    validation_loader: DataLoader


@dataclass(slots=True)
class TrainingComponents:
    """
    Container holding initialized training components.

    Attributes
    ----------
    model : nn.Module
        Neural network model.

    criterion : nn.Module
        Loss function.

    optimizer : Optimizer
        Optimizer used during training.

    scheduler : LRScheduler | None
        Learning rate scheduler.
    """

    model: nn.Module
    criterion: nn.Module
    optimizer: Optimizer
    scheduler: LRScheduler | None