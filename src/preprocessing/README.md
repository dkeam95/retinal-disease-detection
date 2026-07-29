# Preprocessing Module

## Purpose

The `preprocessing` module constructs reproducible image transformation pipelines using Albumentations for training, validation, and testing splits.

It is strictly responsible for converting raw fundus images (NumPy arrays) into normalized PyTorch tensors suitable for model inference.

---

## Pipelines

### Training Pipeline (`build_train_pipeline`)
- Image Resizing ($512 \times 512$ default);
- Configurable Augmentations:
  - Horizontal / Vertical Flips;
  - Random Rotation;
  - Random Brightness / Contrast;
- ImageNet Channel Normalization ($Mean=[0.485, 0.456, 0.406], Std=[0.229, 0.224, 0.225]$);
- PyTorch Tensor Conversion (`ToTensorV2`).

### Validation & Testing Pipelines (`build_validation_pipeline`, `build_test_pipeline`)
- Deterministic Image Resizing ($512 \times 512$);
- ImageNet Channel Normalization;
- PyTorch Tensor Conversion (`ToTensorV2`).

---

## Module Structure

```text
preprocessing/
├── __init__.py      # Public API exports
├── pipeline.py      # Pipeline construction functions
├── transforms.py    # Individual transformation mappings
├── types.py          # PreprocessingConfig dataclasses
├── exceptions.py     # PreprocessingError hierarchy
└── README.md        # Technical documentation
```

---

## Public API Usage

```python
from preprocessing import build_train_pipeline, build_validation_pipeline

train_transform = build_train_pipeline(config.preprocessing)
val_transform = build_validation_pipeline(config.preprocessing)

# Apply to raw NumPy image
transformed = train_transform(image=raw_image)
tensor_image = transformed["image"]
```

---

## Design Principles

- **Separation of Concerns**: Only transforms image data; does not perform data loading, annotation parsing, or batch collation.
- **Configuration Driven**: All augmentation parameters (probabilities, angles, crop dimensions) are driven by YAML configurations.
