# Preprocessing Module

## Responsibility

The preprocessing module is responsible only for transforming raw images
into tensors suitable for neural network input.

It does not perform:

- dataset loading;
- annotation parsing;
- model creation;
- training;
- evaluation;
- inference.

---

## Public API

```python
build_train_pipeline(config: PreprocessingConfig) -> A.Compose

build_validation_pipeline(config: PreprocessingConfig) -> A.Compose

build_test_pipeline(config: PreprocessingConfig) -> A.Compose
```

---

## Dependencies

- albumentations
- albumentations.pytorch
- torch
- numpy
- common.config

---

## File Structure

```
preprocessing/
├── __init__.py
├── config.py
├── exceptions.py
├── pipeline.py
├── transforms.py
├── types.py
└── README.md
```

---

## Workflow

```
Raw Image (NumPy)
        │
        ▼
Resize
        │
        ▼
Optional Augmentations
        │
        ├── Horizontal Flip
        ├── Vertical Flip
        ├── Rotation
        └── Brightness / Contrast
        │
        ▼
Normalize
        │
        ▼
ToTensorV2
        │
        ▼
Torch Tensor
```

---

## Pipelines

### Training

```
Resize
    │
    ▼
Horizontal Flip (optional)
    │
    ▼
Vertical Flip (optional)
    │
    ▼
Rotation (optional)
    │
    ▼
Brightness / Contrast (optional)
    │
    ▼
Normalize
    │
    ▼
ToTensorV2
```

### Validation

```
Resize
    │
    ▼
Normalize
    │
    ▼
ToTensorV2
```

### Testing

```
Resize
    │
    ▼
Normalize
    │
    ▼
ToTensorV2
```

---

## Design Principles

- Single Responsibility Principle
- Configuration-driven preprocessing
- Reusable transformations
- No duplicated pipelines
- Strong typing
- Easy extension with new augmentations
