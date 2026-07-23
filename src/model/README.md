# Model Module

## Overview

The `model` module is responsible for constructing neural network architectures used for diabetic retinopathy classification.

The module follows a configuration-driven, registry-based, and modular architecture. Models are assembled from independent components rather than relying on monolithic implementations. This design makes the project scalable, maintainable, and easy to extend with new architectures.

---

## Responsibilities

The model module is responsible for:

- Building neural network architectures.
- Constructing feature extraction backbones.
- Constructing classification heads.
- Combining backbone and classifier into a complete model.
- Registering supported model architectures.
- Creating models from configuration.
- Providing strongly typed model outputs.
- Defining model-specific exceptions.

---

## Architecture

```
ModelConfig
      │
      ▼
ModelFactory
      │
      ▼
ModelRegistry
      │
      ▼
ModelBuilder
      │
      ├──────────────┐
      ▼              ▼
 Backbone      Classifier
      │              │
      └──────┬───────┘
             ▼
 ClassificationModel
             │
             ▼
        ModelOutput
```

---

## Module Structure

```
model/
│
├── backbone.py
├── builder.py
├── classifier.py
├── classification_model.py
├── exceptions.py
├── factory.py
├── model_names.py
├── registry.py
├── types.py
└── README.md
```

---

## Components

### backbone.py

Constructs feature extraction backbones using TIMM.

The backbone is created with:

```python
num_classes=0
```

which removes the default classifier and returns feature vectors instead of classification logits.

---

### classifier.py

Builds the classification head.

The current implementation consists of:

- Dropout
- Fully Connected Layer

The classifier is independent of the backbone and can be replaced without modifying the rest of the architecture.

---

### classification_model.py

Combines the backbone and classifier into a single PyTorch model.

Forward pipeline:

```
Input Image
      │
      ▼
Backbone
      │
      ▼
Feature Vector
      │
      ▼
Classifier
      │
      ▼
Logits
```

The forward method returns a `ModelOutput` object instead of raw tensors.

---

### builder.py

Constructs complete classification models by assembling:

- Backbone
- Classifier
- ClassificationModel

The builder is architecture-independent and supports any registered TIMM model.

---

### registry.py

Stores the mapping between model architectures and builder functions.

Example:

```python
ModelArchitecture.EFFICIENTNET_B0
        ↓
build_timm_model()
```

Adding support for a new architecture only requires updating this registry.

---

### factory.py

The only public entry point for creating models.

All training and inference code should use:

```python
create_model(config)
```

instead of instantiating models directly.

---

### model_names.py

Defines the `ModelArchitecture` enumeration using `StrEnum`.

This prevents invalid architecture names while improving IDE autocomplete and static type checking.

---

### types.py

Defines common model data structures.

Currently:

```python
ModelOutput
```

contains:

- logits
- features

This design allows future support for feature extraction, explainability, and metric learning.

---

### exceptions.py

Defines model-specific exceptions.

Current hierarchy:

- ModelError
- UnknownModelArchitectureError
- ModelInitializationError

---

## Current Supported Architectures

Currently implemented:

- EfficientNet-B0

Planned support:

- EfficientNet-B3
- EfficientNet-B4
- ConvNeXt
- ResNet
- Vision Transformer (ViT)
- Swin Transformer

---

## Public API

```python
from model.factory import create_model

model = create_model(config)
```

No other module should instantiate models directly.

---

## Tests

```
tests/model/
│
├── test_backbone.py
├── test_builder.py
├── test_classifier.py
├── test_classification_model.py
├── test_exceptions.py
├── test_model_names.py
└── test_types.py
```

---

## Design Principles

The model module follows several architectural principles:

- Single Responsibility Principle.
- Configuration-driven design.
- Registry-based model selection.
- Factory pattern for model creation.
- Modular backbone/classifier separation.
- Strong typing.
- Extensible architecture.
- Production-oriented design.

---

## Extending the Module

To add a new backbone:

1. Add the architecture to `ModelArchitecture`.
2. Register it in `MODEL_REGISTRY`.
3. No changes to the factory are required.
4. Add unit tests.

---

## Future Improvements

The current architecture is intentionally designed for future expansion.

Planned improvements include:

- YAML-configurable classifier heads.
- Configurable dropout.
- Backbone freezing.
- Fine-tuning strategies.
- Feature extraction API.
- Grad-CAM support.
- Metric Learning.
- ArcFace.
- Multi-head classification.
- Knowledge Distillation.
- ONNX and TorchScript export.

---

## Data Flow

```
Input Image
      │
      ▼
Backbone
      │
      ▼
Feature Vector
      │
      ▼
Classifier
      │
      ▼
Logits
      │
      ▼
ModelOutput
```

The separation between backbone and classifier provides maximum flexibility for future experimentation while keeping the training pipeline independent from any specific neural network architecture.
