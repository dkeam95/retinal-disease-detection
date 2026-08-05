"""
Training and validation steps module.

This module contains functions for processing a single batch during training
and validation, including forward passes, loss calculation, backward passes,
and model weight updates.
"""

from __future__ import annotations  # Enables modern type hints

import torch  # PyTorch core library
from torch import Tensor, nn  # Base classes for tensors and neural network modules
from torch.optim import Optimizer  # Base class for optimization algorithms

from trainer.types import (
    StepOutput,  # Container storing loss value and batch size for one step
)
from trainer.utils import (
    detach_loss,  # Utility function to convert loss tensor to a python float
)


def train_step(
    model: nn.Module,
    optimizer: Optimizer,
    criterion: nn.Module,
    images: Tensor,
    targets: Tensor,
    scaler: torch.amp.GradScaler | None = None,
    mixed_precision: bool = False,
    device_type: str = "cuda",
) -> StepOutput:
    """
    Execute one full training step on a single batch of data with optional AMP.
    """

    optimizer.zero_grad(set_to_none=True)

    with torch.amp.autocast(device_type=device_type, enabled=mixed_precision):
        outputs = model(images)
        logits_tensor = outputs.logits if hasattr(outputs, "logits") else outputs
        loss = criterion(logits_tensor, targets)

    if scaler is not None and mixed_precision:
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        optimizer.step()

    return StepOutput(
        loss=detach_loss(loss),
        batch_size=targets.size(0),
    )


def validation_step(
    model: nn.Module,
    criterion: nn.Module,
    images: Tensor,
    targets: Tensor,
) -> tuple[Tensor, StepOutput]:
    """
    Execute one evaluation step on a single batch of validation data.

    Parameters
    ----------
    model : nn.Module
        Neural network model instance.
    criterion : nn.Module
        Loss function module.
    images : Tensor
        Input batch of images.
    targets : Tensor
        Ground truth labels for the batch.

    Returns
    -------
    tuple[Tensor, StepOutput]
        A tuple containing predicted model logits and the step output statistics.
    """

    # Disable gradient tracking to reduce memory usage and speed up evaluation
    with torch.no_grad():
        # Pass the input batch through the model to get predictions
        outputs = model(images)
        logits_tensor = outputs.logits if hasattr(outputs, "logits") else outputs

        # Calculate the validation loss value
        loss = criterion(logits_tensor, targets)

    # Return model predictions alongside loss and batch size statistics
    return logits_tensor, StepOutput(
        loss=detach_loss(loss),
        batch_size=targets.size(0),
    )
