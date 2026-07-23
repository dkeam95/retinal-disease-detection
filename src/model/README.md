# Model Module

## Overview

The `model` module is responsible for constructing neural network architectures used for diabetic retinopathy classification.

The module follows a configuration-driven and modular architecture, allowing different backbone networks to be selected without modifying the training pipeline.

---

## Responsibilities

- Register supported model architectures.
- Build neural network models.
- Instantiate models from configuration.
- Validate requested model architectures.
- Provide model-specific exceptions.
- Serve as the entry point for model creation.

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
      ▼
PyTorch Model
```

---

## Module Structure

```
model/
│
├── builder.py
├── exceptions.py
├── factory.py
├── model_names.py
├── registry.py
└── README.md
```

---

## Components

### builder.py

Responsible for constructing PyTorch models.

Currently provides a generic TIMM builder capable of creating any supported TIMM architecture.

---

### factory.py

Creates neural network models from a `ModelConfig`.

The factory is the only public entry point used by the training pipeline.

---

### registry.py

Stores the mapping between model architecture names and their corresponding builder functions.

Adding support for a new model only requires registering it in this file.

---

### model_names.py

Contains the `ModelArchitecture` enumeration.

Using `StrEnum` prevents invalid architecture names and improves IDE autocomplete and static type checking.

---

### exceptions.py

Defines all model-specific exceptions.

Current exceptions:

- `ModelError`
- `UnknownModelArchitectureError`
- `ModelInitializationError`

---

## Current Supported Architectures

- EfficientNet-B0

Future planned architectures:

- EfficientNet-B3
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

The rest of the project should never instantiate models directly.

---

## Tests

```
tests/model/
│
├── test_builder.py
├── test_exceptions.py
└── test_model_names.py
```

---

## Design Principles

- Single Responsibility Principle.
- Configuration-driven architecture.
- Registry-based model selection.
- Strong typing using `StrEnum`.
- Centralized model creation through a factory.
- Easily extensible without modifying existing code.

---

## Extending the Module

To add a new architecture:

1. Add a new value to `ModelArchitecture`.
2. Register the architecture in `MODEL_REGISTRY`.
3. No changes to the factory are required.
4. Add unit tests for the new architecture.

---

## Future Improvements

The current implementation creates complete TIMM classification models.

In future iterations the architecture will evolve into:

```
Backbone
      │
      ▼
Feature Extractor
      │
      ▼
Classification Head
```

This separation will support:

- Fine-tuning
- Feature extraction
- Grad-CAM
- Custom classification heads
- Metric learning
- ArcFace
- Flexible transfer learning
