# Preprocessing Module

## Purpose

The preprocessing module is responsible for transforming raw dataset
samples into model-ready inputs.

This module performs image preprocessing and data augmentation while
remaining independent from the dataset, model, and training pipeline.

---

## Responsibilities

- Build preprocessing pipelines.
- Apply image augmentations.
- Normalize images.
- Convert images to tensors.
- Provide separate pipelines for training, validation, and testing.

---

## Out of Scope

This module does NOT:

- Load images from disk.
- Read dataset annotations.
- Train models.
- Create DataLoaders.
- Perform inference.

---

## Public API

Planned public interface:

- build_train_pipeline()
- build_validation_pipeline()
- build_test_pipeline()

---

## Dependencies

- Albumentations
- OpenCV
- NumPy
- PyTorch

---

## Status

- [x] Module initialized
- [ ] Configuration
- [ ] Pipeline implementation
- [ ] Custom transforms
- [ ] Tests
