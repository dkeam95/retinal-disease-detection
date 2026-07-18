# Model Module

## Purpose

The model module is responsible for constructing neural network models
from a configuration object.

---

## Responsibilities

This module is responsible only for:

- creating neural network models;
- selecting the requested architecture;
- returning an initialized `torch.nn.Module`.

This module does **not** perform:

- dataset loading;
- image preprocessing;
- training;
- evaluation;
- inference;
- checkpoint management;
- optimizer creation;
- scheduler creation.

---

## Public API

```python
create_model(config: ModelConfig) -> torch.nn.Module
```

---

## Input

```text
ModelConfig
```

---

## Output

```text
torch.nn.Module
```

---

## Dependencies

- torch
- timm
- common.config

---

## File Structure

```text
model/
├── __init__.py
├── builder.py
├── factory.py
├── registry.py
└── README.md
```

---

## Workflow

```text
ModelConfig
      │
      ▼
create_model()
      │
      ▼
MODEL_REGISTRY
      │
      ▼
builder()
      │
      ▼
torch.nn.Module
```

---

## Supported Architectures

### Current

- EfficientNet-B0

### Planned

- EfficientNet family
- ResNet family
- DenseNet family
- ConvNeXt family
- Vision Transformer (ViT) family
- Swin Transformer family

The module is designed so that new architectures can be added without modifying the public API or the model factory.

---

## Design Principles

The module follows the following design principles:

- Single Responsibility Principle (SRP)
- Configuration-driven model creation
- Registry pattern
- Factory pattern
- Easy architecture replacement
- Extensible design
- Minimal coupling between modules

---

## Future Extensions

Future versions of this module may include:

- loading pretrained checkpoints;
- automatic model registration;
- custom architectures;
- feature extractor builders;
- layer freezing utilities;
- backbone replacement utilities;
- support for custom classification heads.
