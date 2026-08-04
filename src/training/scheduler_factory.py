"""Scheduler factory.

This module provides a factory for constructing learning rate schedulers.
"""

from __future__ import annotations

from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler, OneCycleLR, StepLR

from training.exceptions import SchedulerFactoryError
from training.utils import normalize_name, validate_component_name

_SUPPORTED_SCHEDULERS: tuple[str, ...] = (
    "cosine",
    "onecycle",
    "step",
)


class SchedulerFactory:
    """Factory for creating learning rate schedulers."""

    @staticmethod
    def build(
        name: str,
        optimizer: Optimizer,
        *,
        epochs: int,
        steps_per_epoch: int | None = None,
        step_size: int = 30,
        gamma: float = 0.1,
        max_learning_rate: float | None = None,
    ) -> LRScheduler:
        """
        Build a learning rate scheduler.

        Parameters
        ----------
        name : str
            Scheduler name.

        optimizer : Optimizer
            Optimizer instance.

        epochs : int
            Number of training epochs.

        steps_per_epoch : int | None, default=None
            Number of optimizer updates per epoch.

        step_size : int, default=30
            Step interval for StepLR.

        gamma : float, default=0.1
            Learning rate decay factor.

        max_learning_rate : float | None, default=None
            Maximum learning rate for OneCycleLR.

        Returns
        -------
        LRScheduler
            Initialized scheduler.

        Raises
        ------
        SchedulerFactoryError
            If scheduler creation fails.
        """

        try:
            validate_component_name(name=name, supported=_SUPPORTED_SCHEDULERS)

        except ValueError as error:
            raise SchedulerFactoryError(str(error)) from error

        name = normalize_name(name)

        if name == "cosine":
            return CosineAnnealingLR(optimizer, T_max=epochs)

        if name == "step":
            return StepLR(optimizer, step_size=step_size, gamma=gamma)

        if name == "onecycle":
            if steps_per_epoch is None:
                raise SchedulerFactoryError(
                    "steps_per_epoch is required for OneCycleLR."
                )

            if max_learning_rate is None:
                raise SchedulerFactoryError(
                    "max_learning_rate is required for OneCycleLR."
                )

            return OneCycleLR(
                optimizer,
                max_lr=max_learning_rate,
                steps_per_epoch=steps_per_epoch,
                epochs=epochs,
            )

        raise SchedulerFactoryError(f"Unknown scheduler name: {name}")
