# Model Module

## Responsibility

The model module is responsible only for creating neural network models.

It does not perform:

- dataset loading;
- image preprocessing;
- training;
- evaluation;
- inference;
- checkpoint management.

---

## Public API

```python
create_model(config: ModelConfig) -> torch.nn.Module
```

---

## Dependencies

- torch
- timm
- common.config

---

## File Structure

```
model/
├── __init__.py
├── factory.py
└── README.md
```

---

## Workflow

```
ModelConfig
      │
      ▼
create_model()
      │
      ▼
torch.nn.Module
```

---

## Supported Architectures

Initially the module will support:

- EfficientNet-B0

The architecture is designed for future extension:

- EfficientNet family
- ResNet family
- ConvNeXt family
- Vision Transformer family

without changing the public API.

---

## Design Principles

- Single Responsibility Principle
- Configuration-driven model creation
- Framework-independent project architecture
- Easy model replacement
