"""
Model factory.

This module provides a factory for constructing neural network models used
during training.
"""

from __future__ import annotations

import torch.nn as nn
from torchvision.models import (
    EfficientNet_B0_Weights,
    efficientnet_b0,
)

from training.exceptions import ModelFactoryError
from training.utils import (
    normalize_name,
    validate_component_name,
)


_SUPPORTED_MODELS: tuple[str, ...] = (
    "efficientnet_b0",
)


class ModelFactory:
    """
    Factory for constructing neural network models.
    """

    @staticmethod
    def build(
        architecture: str,
        *,
        pretrained: bool,
        num_classes: int,
    ) -> nn.Module:
        """
        Build a neural network model.

        Parameters
        ----------
        architecture : str
            Model architecture name.

        pretrained : bool
            Whether to load pretrained weights.

        num_classes : int
            Number of output classes.

        Returns
        -------
        nn.Module
            Initialized neural network.

        Raises
        ------
        ModelFactoryError
            If the requested architecture is not supported.
        """

        try:
            validate_component_name(
                name=architecture,
                supported=_SUPPORTED_MODELS,
            )
        except ValueError as error:
            raise ModelFactoryError(
                str(error),
            ) from error

        normalized_architecture = normalize_name(
            architecture,
        )

        if normalized_architecture == "efficientnet_b0":
            return ModelFactory._build_efficientnet_b0(
                pretrained=pretrained,
                num_classes=num_classes,
            )

        raise ModelFactoryError(
            f"Unknown model architecture '{architecture}'."
        )

    @staticmethod
    def _build_efficientnet_b0(
        *,
        pretrained: bool,
        num_classes: int,
    ) -> nn.Module:
        """
        Build an EfficientNet-B0 model.

        Parameters
        ----------
        pretrained : bool
            Whether to load ImageNet pretrained weights.

        num_classes : int
            Number of output classes.

        Returns
        -------
        nn.Module
            Configured EfficientNet-B0 model.
        """

        weights = (
            EfficientNet_B0_Weights.DEFAULT
            if pretrained
            else None
        )

        model = efficientnet_b0(
            weights=weights,
        )

        ModelFactory._replace_classifier(
            model=model,
            num_classes=num_classes,
        )

        return model

    @staticmethod
    def _replace_classifier(
        model: nn.Module,
        num_classes: int,
    ) -> None:
        """
        Replace the classification head of an EfficientNet model.

        Parameters
        ----------
        model : nn.Module
            EfficientNet model.

        num_classes : int
            Number of output classes.

        Raises
        ------
        ModelFactoryError
            If the classifier structure is unexpected.
        """

        if not isinstance(
            model.classifier,
            nn.Sequential,
        ):
            raise ModelFactoryError(
                "Unexpected EfficientNet classifier structure."
            )

        last_layer = model.classifier[-1]

        if not isinstance(
            last_layer,
            nn.Linear,
        ):
            raise ModelFactoryError(
                "Expected the last classifier layer to be nn.Linear."
            )

        in_features = last_layer.in_features

        model.classifier[-1] = nn.Linear(
            in_features=in_features,
            out_features=num_classes,
        )

        ModelFactory._initialize_classifier(
            model.classifier[-1],
        )

    @staticmethod
    def _initialize_classifier(
        classifier: nn.Linear,
    ) -> None:
        """
        Initialize the classifier layer.

        Parameters
        ----------
        classifier : nn.Linear
            Classification layer.
        """

        nn.init.xavier_uniform_(
            classifier.weight,
        )

        if classifier.bias is not None:
            nn.init.zeros_(
                classifier.bias,
            )

    @staticmethod
    def count_parameters(
        model: nn.Module,
    ) -> tuple[int, int]:
        """
        Count total and trainable model parameters.

        Parameters
        ----------
        model : nn.Module
            Neural network model.

        Returns
        -------
        tuple[int, int]
            Tuple containing:

            - total parameters
            - trainable parameters
        """

        total_parameters = sum(
            parameter.numel()
            for parameter in model.parameters()
        )

        trainable_parameters = sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )

        return total_parameters, trainable_parameters