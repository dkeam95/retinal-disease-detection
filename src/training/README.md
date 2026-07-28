# Training Module

## Overview

The `training` module is responsible for constructing and configuring all
core components required for neural network training.

The module follows the **Factory Pattern**, allowing every training component
to be created from configuration files while keeping the training loop
independent from implementation details.

The goal of this module is to provide a clean, modular, and reproducible
training pipeline.

---

# Module Structure

```text
training/

├── __init__.py
├── exceptions.py
├── types.py
├── utils.py
│
├── loss_factory.py
├── optimizer_factory.py
├── scheduler_factory.py
├── model_factory.py
│
└── README.md
```

---

# Components

## ModelFactory

Responsible for constructing neural network architectures.

### Current supported models

- EfficientNet-B0

### Responsibilities

- Build neural network architecture
- Load ImageNet pretrained weights
- Replace classification head
- Initialize classifier weights
- Count total and trainable parameters

---

## LossFactory

Responsible for constructing loss functions.

### Current supported losses

- CrossEntropyLoss

### Supported features

- Class weighting
- Label smoothing

The factory returns a fully initialized PyTorch loss object.

---

## OptimizerFactory

Responsible for constructing optimizers.

### Current supported optimizers

- Adam
- AdamW
- SGD

The optimizer is created directly from configuration parameters.

---

## SchedulerFactory

Responsible for constructing learning rate schedulers.

### Current supported schedulers

- CosineAnnealingLR
- StepLR
- OneCycleLR

Each scheduler is configured independently from the training loop.

---

# Utilities

The module also provides several helper utilities.

## utils.py

Contains helper functions for:

- component name normalization
- component validation
- common training helpers

---

## exceptions.py

Contains custom exceptions for all training factories.

Examples:

- ModelFactoryError
- LossFactoryError
- OptimizerFactoryError
- SchedulerFactoryError

---

## types.py

Contains common type aliases used throughout the training package.

---

# Design Principles

The module follows several software engineering principles.

## Single Responsibility Principle (SRP)

Each factory is responsible only for constructing one specific object.

- ModelFactory → models
- LossFactory → loss functions
- OptimizerFactory → optimizers
- SchedulerFactory → schedulers

Factories never perform training.

---

## Factory Pattern

Object creation is centralized.

Instead of directly creating PyTorch objects inside the training loop,

```python
model = efficientnet_b0(...)
optimizer = AdamW(...)
scheduler = CosineAnnealingLR(...)
```

the training loop becomes

```python
model = ModelFactory.build(...)
optimizer = OptimizerFactory.build(...)
scheduler = SchedulerFactory.build(...)
```

This greatly simplifies future extensions.

---

## Configuration-driven Design

Every component is created from YAML configuration.

Example:

```yaml
model:
  architecture: efficientnet_b0
  pretrained: true
  num_classes: 5

optimizer:
  name: adamw
  learning_rate: 0.0003
  weight_decay: 0.0001

scheduler:
  name: cosine

loss:
  name: cross_entropy
```

The training pipeline never depends on hardcoded parameters.

---

## Reproducibility

The module is designed to guarantee reproducible experiments.

Configuration files completely describe:

- model architecture
- optimizer
- scheduler
- loss function

This makes experiments easy to reproduce.

---

# Typical Usage

```python
model = ModelFactory.build(
    architecture="efficientnet_b0",
    pretrained=True,
    num_classes=5,
)

criterion = LossFactory.build(
    name="cross_entropy",
)

optimizer = OptimizerFactory.build(
    name="adamw",
    parameters=model.parameters(),
    learning_rate=3e-4,
    weight_decay=1e-4,
)

scheduler = SchedulerFactory.build(
    name="cosine",
    optimizer=optimizer,
    epochs=100,
)
```

The constructed objects are then passed to the `Trainer`.

---

# Future Extensions

The architecture is intentionally designed for future expansion.

## Planned model architectures

- EfficientNet-B3
- ConvNeXt-Tiny
- ResNet-50
- Vision Transformer (ViT)

---

## Planned loss functions

- Focal Loss
- Class Balanced Loss
- Dice Loss

---

## Planned optimizers

- RMSProp
- Lion

---

## Planned schedulers

- CosineAnnealingWarmRestarts
- ReduceLROnPlateau
- PolynomialLR

No changes to the training loop will be required when new components are added.

Only the corresponding Factory will be extended.

---

# Summary

The `training` module provides a modular and extensible interface for
constructing all components required for neural network training.

Its architecture emphasizes:

- modularity
- reproducibility
- maintainability
- configurability
- separation of responsibilities

This design allows the training pipeline to remain clean while supporting
future model architectures, optimizers, schedulers, and loss functions with
minimal code changes.
