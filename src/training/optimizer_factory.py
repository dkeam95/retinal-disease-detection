"""Optimizer factory.

This module provides a factory for constructing optimizer used during
model training.
"""

from __future__ import annotations

from collections.abc import Iterable

from torch import nn
from torch.optim import SGD, Adam, AdamW, Optimizer

from training.exceptions import OptimizerFactoryError
from training.utils import normalize_name, validate_component_name

_SUPPORTED_OPTIMIZERS: tuple[str, ...] = (
    "adam",
    "adamw",
    "sgd",
)

class OptimizerFactory:
    """Factory for creating optimizers."""

    @staticmethod
    def build(
        name: str,
        parameters: Iterable[nn.Parameter],
        *,
        learning_rate: float,
        weight_decay: float = 0.0,
        momentum: float = 0.9
    ) -> Optimizer:
        """
        Build an optimizer.

        Parameters
        ----------
        name : str
            Optimizer name.

        parameters : Iterable[nn.Parameter]
            Model parameters.

        learning_rate : float
            Learning rate.

        weight_decay : float, default=0.0
            Weight decay coefficient.

        momentum : float, default=0.9
            SGD momentum.

        Returns
        -------
        Optimizer
            Initialized optimizer.

        Raises
        ------
        OptimizerFactoryError
            If the optimizer is not supported.
        """

        try:
            validate_component_name(
                name=name,
                supported=_SUPPORTED_OPTIMIZERS
            )

        except ValueError as error:
            raise OptimizerFactoryError(str(error)) from error

        name = normalize_name(name)

        if name == "adam":
            return Adam(
                parameters,
                lr=learning_rate,
                weight_decay=weight_decay
            )

        if name == "adamw":
            return AdamW(
                parameters,
                lr=learning_rate,
                weight_decay=weight_decay,
            )

        if name == "sgd":
            return SGD(
                parameters,
                lr=learning_rate,
                weight_decay=weight_decay,
                momentum=momentum
            )

        raise OptimizerFactoryError(f"Unsupported optimizer: {name}")
