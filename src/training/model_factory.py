"""
Model factory.

This module provides a factory for constructing neural network models used
during training.
"""

from __future__ import annotations

import torch.nn as nn
from torchvision.models import (
    ConvNeXt_Tiny_Weights,
    DenseNet121_Weights,
    EfficientNet_B0_Weights,
    MobileNet_V3_Large_Weights,
    ResNet18_Weights,
    ResNet50_Weights,
    Swin_T_Weights,
    ViT_B_16_Weights,
    convnext_tiny,
    densenet121,
    efficientnet_b0,
    mobilenet_v3_large,
    resnet18,
    resnet50,
    swin_t,
    vit_b_16,
)

from training.exceptions import ModelFactoryError
from training.utils import (
    normalize_name,
    validate_component_name,
)

_SUPPORTED_MODELS: tuple[str, ...] = (
    "efficientnet_b0",
    "resnet18",
    "resnet50",
    "densenet121",
    "mobilenet_v3_large",
    "convnext_tiny",
    "vit_b_16",
    "swin_t",
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
        dropout_rate: float = 0.0,
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

        builders = {
            "efficientnet_b0": ModelFactory._build_efficientnet_b0,
            "resnet18": ModelFactory._build_resnet18,
            "resnet50": ModelFactory._build_resnet50,
            "densenet121": ModelFactory._build_densenet121,
            "mobilenet_v3_large": ModelFactory._build_mobilenet_v3_large,
            "convnext_tiny": ModelFactory._build_convnext_tiny,
            "vit_b_16": ModelFactory._build_vit_b_16,
            "swin_t": ModelFactory._build_swin_t,
        }

        builder = builders.get(normalized_architecture)
        if builder is None:
            raise ModelFactoryError(f"Unknown model architecture '{architecture}'.")

        model = builder(pretrained=pretrained, num_classes=num_classes)
        if dropout_rate > 0.0:
            ModelFactory._add_dropout(model, dropout_rate)
        return model

    @staticmethod
    def _add_dropout(model: nn.Module, dropout_rate: float) -> None:
        """Add dropout before final classification layer."""
        if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
            in_features = model.fc.in_features
            out_features = model.fc.out_features
            model.fc = nn.Sequential(
                nn.Dropout(p=dropout_rate), nn.Linear(in_features, out_features)
            )
        elif hasattr(model, "classifier"):
            if isinstance(model.classifier, nn.Linear):
                in_features = model.classifier.in_features
                out_features = model.classifier.out_features
                model.classifier = nn.Sequential(
                    nn.Dropout(p=dropout_rate), nn.Linear(in_features, out_features)
                )
            elif isinstance(model.classifier, nn.Sequential):
                last_layer = model.classifier[-1]
                if isinstance(last_layer, nn.Linear):
                    in_features = last_layer.in_features
                    out_features = last_layer.out_features
                    model.classifier[-1] = nn.Sequential(
                        nn.Dropout(p=dropout_rate), nn.Linear(in_features, out_features)
                    )
        elif (
            hasattr(model, "heads")
            and hasattr(model.heads, "head")
            and isinstance(model.heads.head, nn.Linear)
        ):
            in_features = model.heads.head.in_features
            out_features = model.heads.head.out_features
            model.heads.head = nn.Sequential(
                nn.Dropout(p=dropout_rate), nn.Linear(in_features, out_features)
            )
        elif hasattr(model, "head") and isinstance(model.head, nn.Linear):
            in_features = model.head.in_features
            out_features = model.head.out_features
            model.head = nn.Sequential(
                nn.Dropout(p=dropout_rate), nn.Linear(in_features, out_features)
            )

    # ------------------------------------------------------------------
    # Individual model builders
    # ------------------------------------------------------------------

    @staticmethod
    def _build_efficientnet_b0(
        *,
        pretrained: bool,
        num_classes: int,
    ) -> nn.Module:
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = efficientnet_b0(weights=weights)
        ModelFactory._replace_sequential_classifier(
            model=model, num_classes=num_classes
        )
        return model

    @staticmethod
    def _build_resnet18(
        *,
        pretrained: bool,
        num_classes: int,
    ) -> nn.Module:
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        ModelFactory._initialize_classifier(model.fc)
        return model

    @staticmethod
    def _build_resnet50(
        *,
        pretrained: bool,
        num_classes: int,
    ) -> nn.Module:
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        model = resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        ModelFactory._initialize_classifier(model.fc)
        return model

    @staticmethod
    def _build_densenet121(
        *,
        pretrained: bool,
        num_classes: int,
    ) -> nn.Module:
        weights = DenseNet121_Weights.DEFAULT if pretrained else None
        model = densenet121(weights=weights)
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
        ModelFactory._initialize_classifier(model.classifier)
        return model

    @staticmethod
    def _build_mobilenet_v3_large(
        *,
        pretrained: bool,
        num_classes: int,
    ) -> nn.Module:
        weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        model = mobilenet_v3_large(weights=weights)
        # MobileNetV3 classifier is Sequential: [Linear, Hardswish, Dropout, Linear]
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        ModelFactory._initialize_classifier(model.classifier[-1])
        return model

    @staticmethod
    def _build_convnext_tiny(
        *,
        pretrained: bool,
        num_classes: int,
    ) -> nn.Module:
        weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        model = convnext_tiny(weights=weights)
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_features, num_classes)
        ModelFactory._initialize_classifier(model.classifier[2])
        return model

    @staticmethod
    def _build_vit_b_16(
        *,
        pretrained: bool,
        num_classes: int,
    ) -> nn.Module:
        weights = ViT_B_16_Weights.DEFAULT if pretrained else None
        model = vit_b_16(weights=weights)
        # ViT classifier head: model.heads.head is nn.Linear(768, 1000)
        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, num_classes)
        ModelFactory._initialize_classifier(model.heads.head)
        return model

    @staticmethod
    def _build_swin_t(
        *,
        pretrained: bool,
        num_classes: int,
    ) -> nn.Module:
        weights = Swin_T_Weights.DEFAULT if pretrained else None
        model = swin_t(weights=weights)
        # Swin Transformer classifier: model.head is nn.Linear(768, 1000)
        in_features = model.head.in_features
        model.head = nn.Linear(in_features, num_classes)
        ModelFactory._initialize_classifier(model.head)
        return model

    # ------------------------------------------------------------------
    # Helper utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _replace_sequential_classifier(
        model: nn.Module,
        num_classes: int,
    ) -> None:
        if not isinstance(model.classifier, nn.Sequential):
            raise ModelFactoryError("Unexpected EfficientNet classifier structure.")

        last_layer = model.classifier[-1]
        if not isinstance(last_layer, nn.Linear):
            raise ModelFactoryError(
                "Expected the last classifier layer to be nn.Linear."
            )

        in_features = last_layer.in_features
        model.classifier[-1] = nn.Linear(
            in_features=in_features, out_features=num_classes
        )
        ModelFactory._initialize_classifier(model.classifier[-1])

    @staticmethod
    def _initialize_classifier(
        classifier: nn.Linear,
    ) -> None:
        nn.init.xavier_uniform_(classifier.weight)
        if classifier.bias is not None:
            nn.init.zeros_(classifier.bias)

    @staticmethod
    def count_parameters(
        model: nn.Module,
    ) -> tuple[int, int]:
        total_parameters = sum(parameter.numel() for parameter in model.parameters())
        trainable_parameters = sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
        return total_parameters, trainable_parameters
