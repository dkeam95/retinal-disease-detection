# Losses Module

## Overview

The `losses` module provides a modular and extensible interface for
building loss functions used during model training.

The module follows the **Factory + Registry** design pattern,
allowing new loss functions to be added without modifying
the training pipeline.

---

## Supported Loss Functions

| Loss                      | Description                                                                               |
| ------------------------- | ----------------------------------------------------------------------------------------- |
| Cross Entropy             | Standard multi-class classification loss.                                                 |
| Weighted Cross Entropy    | Cross Entropy with class weights for imbalanced datasets.                                 |
| Focal Loss                | Reduces the contribution of easy samples and focuses learning on difficult examples.      |
| Class-Balanced Focal Loss | Combines effective-number class weighting with Focal Loss for highly imbalanced datasets. |

---

## Module Structure

```text
losses/

├── __init__.py
├── factory.py
├── registry.py

├── cross_entropy.py
├── weighted_cross_entropy.py
├── focal.py
├── class_balanced_focal.py
```

---

## Architecture

```text
LossConfig
      │
      ▼
build_loss()
      │
      ▼
LOSS_REGISTRY
      │
      ▼
Builder Function
      │
      ▼
nn.Module
```

The training pipeline never directly creates loss functions.
Instead, it always uses:

```python
criterion = build_loss(
    config.loss,
    class_weights,
)
```

---

## Adding a New Loss Function

1. Implement a new loss module.

```text
losses/my_loss.py
```

2. Implement a builder function.

```python
build_my_loss(
    config,
    class_weights,
)
```

3. Register the builder.

```python
LOSS_REGISTRY["my_loss"] = build_my_loss
```

No other code changes are required.

---

## Testing

All loss implementations include unit tests covering:

- forward computation
- reduction modes
- gradient computation
- invalid configuration handling

Run tests:

```bash
pytest tests/losses
```

---

## Design Principles

- Single Responsibility Principle (SRP)
- Open/Closed Principle (OCP)
- Factory Pattern
- Registry Pattern
- Strong typing
- Fully testable components
