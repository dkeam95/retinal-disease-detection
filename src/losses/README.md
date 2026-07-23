# Losses Module

## Purpose

The `losses` module provides a unified interface for constructing and managing loss functions used during model training.

All loss functions follow a common factory/registry architecture, making them interchangeable through configuration without modifying the training pipeline.

---

## Architecture

```
losses/
│
├── __init__.py
│
├── loss_names.py
├── exceptions.py
│
├── registry.py
├── factory.py
│
├── cross_entropy.py
├── weighted_cross_entropy.py
├── focal.py
├── class_balanced_focal.py
│
└── README.md
```

---

## Supported Loss Functions

### Cross Entropy

Standard multi-class cross entropy.

Configuration:

```yaml
loss:
  name: cross_entropy
```

---

### Weighted Cross Entropy

Cross entropy with manually provided class weights.

Configuration:

```yaml
loss:
  name: weighted_cross_entropy
```

Requires:

- class weights

---

### Focal Loss

Implementation based on:

> Lin et al.
> _Focal Loss for Dense Object Detection_

Designed for highly imbalanced datasets.

Configuration:

```yaml
loss:
  name: focal

gamma: 2.0
alpha: 0.25
```

---

### Class-Balanced Focal Loss

Implementation based on:

> Cui et al.
> _Class-Balanced Loss Based on Effective Number of Samples_

Combines:

- class-balanced weighting
- focal loss

Configuration:

```yaml
loss:
  name: class_balanced_focal

gamma: 2.0
```

Requires:

- class weights

---

## Factory Pattern

Losses are never instantiated directly.

Always use

```python
loss = build_loss(config)
```

The factory automatically selects the correct implementation through the registry.

---

## Registry

Available losses are stored inside

```
LOSS_REGISTRY
```

using

```
LossName
```

instead of raw strings.

This provides:

- IDE autocompletion
- type safety
- compile-time validation
- consistent architecture

---

## Exceptions

The module defines custom exceptions.

```
LossError
```

Base exception.

```
UnknownLossError
```

Raised when an unsupported loss is requested.

```
LossInitializationError
```

Raised when loss configuration is invalid.

---

## Testing

Covered by unit tests:

- loss names
- exceptions
- factory
- cross entropy
- weighted cross entropy
- focal loss
- class-balanced focal loss

All loss builders are validated independently.

---

## Design Principles

- Single Responsibility Principle
- Open/Closed Principle
- Registry Pattern
- Factory Pattern
- Configuration Driven
- Type Safe API
- Modular Design
