# Losses Module

## Responsibility

The losses module is responsible only for creating loss functions used
during neural network training.

It does not perform:

- model creation;
- dataset loading;
- image preprocessing;
- training loops;
- metric computation;
- checkpoint management.

---

## Public API

```python
create_loss(config: LossConfig) -> torch.nn.Module
```

---

## Dependencies

- torch
- common.config

---

## File Structure

```text
losses/
├── __init__.py
├── factory.py
├── registry.py
├── cross_entropy.py
├── weighted_cross_entropy.py
├── focal.py
├── class_balanced_focal.py
└── README.md
```

---

## Workflow

```text
LossConfig
      │
      ▼
create_loss()
      │
      ▼
Loss Registry
      │
      ▼
Selected Loss
      │
      ▼
torch.nn.Module
```

---

## Supported Loss Functions

Initially the module supports:

- CrossEntropyLoss
- Weighted CrossEntropyLoss
- Focal Loss
- Class Balanced Focal Loss

The architecture is designed for future extension:

- Label Smoothing Cross Entropy
- Dice Loss
- Tversky Loss
- PolyLoss
- LDAM Loss

without changing the public API.

---

## Design Principles

- Single Responsibility Principle
- Factory Pattern
- Registry Pattern
- Configuration-driven loss creation
- Strong typing
- Easy extension without modifying the Trainer
