"""Loss factory.

This module provides a factory for constructing loss functions used during
model training.
"""

from __future__ import annotations

from torch import Tensor
from torch import nn

from training.exceptions import LossFactoryError
from training.utils import normalize_name, validate_component_name


_SUPPORTED_LOSSES: tuple[str, ...] = (
    "cross_entropy",
)


class LossFactory:
    """Factory for creating loss functions."""

    @staticmethod
    def build(name: str, *, weight: Tensor | None = None, label_smoothing: float = 0.0) -> nn.Module:
        """
        Build a loss function.

        Parameters
        ----------
        name : str
            Loss function name.

        weight : Tensor | None, optional
            Optional class weights.

        label_smoothing : float, default=0.0
            Label smoothing factor.

        Returns
        -------
        nn.Module
            Initialized loss function.

        Raises
        ------
        LossFactoryError
            If the requested loss function is not supported.
        """

        try:
            validate_component_name(
                name=name,
                supported=_SUPPORTED_LOSSES
            )
        except ValueError as error:
            raise LossFactoryError(str(error)) from error

        name = normalize_name(name)

        if name == "cross_entropy":
            return nn.CrossEntropyLoss(
                weight=weight,
                label_smoothing=label_smoothing
            )

        raise LossFactoryError(f"Unsupported loss function: '{name}'")