"""
Utility functions for the trainer module.

This module provides standalone helper utilities for moving tensors across computing devices,
toggling PyTorch execution modes (train vs. eval), extracting detached scalar losses, and
computing safe average metrics across batch iterations.
"""

from __future__ import annotations  # Enables modern type hints (Python 3.7+)

import torch  # PyTorch tensor library
from torch import Tensor, nn  # PyTorch base tensor class and neural network modules


def move_to_device(
    tensor: Tensor,
    device: str | torch.device,
) -> Tensor:
    """
    Move a tensor to the requested computing device.

    Parameters
    ----------
    tensor : Tensor
        Input PyTorch tensor.
    device : str | torch.device
        Target computing device identifier or `torch.device` object.

    Returns
    -------
    Tensor
        Tensor located on the target device.
    """

    return tensor.to(device)


def set_train_mode(model: nn.Module) -> None:
    """
    Switch a neural network model to training mode.

    Parameters
    ----------
    model : nn.Module
        Neural network model instance.
    """

    model.train()


def set_eval_mode(model: nn.Module) -> None:
    """
    Switch a neural network model to evaluation mode.

    Parameters
    ----------
    model : nn.Module
        Neural network model instance.
    """

    model.eval()


def detach_loss(loss: Tensor) -> float:
    """
    Safely detach a scalar loss tensor from the computation graph and convert it to a Python float.

    Parameters
    ----------
    loss : Tensor
        Computed scalar loss tensor.

    Returns
    -------
    float
        Detached scalar loss value.
    """

    return float(loss.detach().cpu().item())


def calculate_average_loss(
    total_loss: float,
    batches: int,
) -> float:
    """
    Calculate the average loss across all processed batches with zero-division safety.

    Parameters
    ----------
    total_loss : float
        Sum of loss values across processed batches.
    batches : int
        Total number of processed batches.

    Returns
    -------
    float
        Average loss per batch, or 0.0 if no batches were processed.
    """

    if batches == 0:
        return 0.0

    return total_loss / batches
