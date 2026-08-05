# Model Module (`src/model/`)

## Overview

The `model` module manages neural network architecture assembly, backbone instantiation, and customized classifier head construction. It decouples feature extraction from class prediction, allowing easy swap-ins of backbones from the `timm` library.

## Key Features

- **Backbone Decoupling**: Backbones are initialized with `num_classes=0` to act strictly as feature extractors.
- **Custom Classifier Heads**: Standard classification heads consist of a linear layer, customizable dropout, and optional normalization.
- **Structured Outputs**: Forward passes return a typed `ModelOutput` containing both logits and intermediate feature embeddings.

## API & Interfaces

### Classes

#### `ClassificationModel(nn.Module)`
Main model assembly.
- **`__init__(backbone: nn.Module, head: nn.Module)`**: Binds the backbone and classification head.
- **`forward(x: torch.Tensor) -> ModelOutput`**: Performs forward pass.

#### `ModelOutput`
Immutable output container:
- `logits`: `torch.Tensor` of shape `(N, num_classes)`
- `features`: `torch.Tensor` of shape `(N, feature_dim)`

### Functions

- **`create_model(config: ModelConfig) -> ClassificationModel`**
  Primary factory function that builds and loads TIMM models based on configuration inputs.
- **`build_timm_model(arch_name: str, pretrained: bool) -> nn.Module`**
  Helper function loading raw TIMM architectures.

### Exceptions

- `UnknownModelArchitectureError`: Raised when the requested TIMM architecture is not registered or supported.
- `ModelInitializationError`: Raised when weights loading or shape checks fail during initialization.

## Dependencies

- `timm`: Main vision models library.
- `torch`: PyTorch neural networks modules.
