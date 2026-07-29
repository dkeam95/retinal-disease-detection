# Model Module

## Overview

The `model` module constructs neural network architectures for multi-class Diabetic Retinopathy classification on fundus photography.

Following a modular, registry-based design, models are assembled from decoupled components: feature backbones (`backbone.py`) and classification heads (`classifier.py`).

---

## Responsibilities

- Constructing feature extraction backbones via `timm`;
- Building classification heads with configurable dropout and linear layers;
- Assembling backbone and head into a `ClassificationModel`;
- Returning structured `ModelOutput` dataclass (containing logits and feature vectors);
- Managing architecture registry (`MODEL_REGISTRY`) and `ModelArchitecture` enumeration.

---

## Architecture Flow

```text
       ModelConfig
            │
            ▼
      create_model()
            │
            ▼
   ClassificationModel
            │
   ┌────────┴────────┐
   ▼                 ▼
Backbone         Classifier
 (TIMM)            (Head)
   │                 │
   └────────┬────────┘
            ▼
       ModelOutput (logits, features)
```

---

## Module Structure

```text
model/
├── __init__.py               # Public API exports
├── backbone.py               # TIMM backbone feature extractor builder
├── classifier.py             # Classification head builder
├── classification_model.py   # Integrated PyTorch nn.Module
├── builder.py                # Architecture assembly logic
├── factory.py                # Model creation factory entry point
├── model_names.py            # ModelArchitecture enumeration
├── registry.py               # Registry mapping names to builders
├── types.py                  # ModelOutput dataclass
├── exceptions.py             # ModelError hierarchy
└── README.md                 # Technical documentation
```

---

## Supported Architectures

- `efficientnet_b0` (Default baseline)
- Extensible via `MODEL_REGISTRY` for `efficientnet_b3`, `convnext_tiny`, `resnet50`, `vit_base_patch16_224`.

---

## Public API

```python
from model import create_model

# Construct model from configuration
model = create_model(config.model)

# Forward pass returns ModelOutput
output = model(image_tensor)
logits = output.logits
features = output.features
```

---

## Design Principles

- **Separation of Concerns**: Backbones extract features; classifiers compute logits.
- **Structured Output**: Models return `ModelOutput` rather than raw tensors, enabling downstream explainability (e.g., Grad-CAM) and metric learning.
- **Type Safety**: Architectural choices are governed by `ModelArchitecture` enum.
