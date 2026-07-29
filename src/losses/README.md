# Losses Module

## Purpose

The `losses` module provides a unified interface and registry for loss functions used during neural network training.

All loss functions are instantiated via a factory/registry pattern, allowing loss selection directly from configuration without altering training loop code.

---

## Supported Loss Functions

1. **Cross-Entropy Loss (`cross_entropy`)**: Standard multi-class cross-entropy with optional label smoothing.
2. **Weighted Cross-Entropy (`weighted_cross_entropy`)**: Class-weighted cross-entropy for handling imbalanced class distributions.
3. **Focal Loss (`focal`)**: Implements Lin et al. (*Focal Loss for Dense Object Detection*), down-weighting easy examples to focus on hard samples.
4. **Class-Balanced Focal Loss (`class_balanced_focal`)**: Implements Cui et al. (*Class-Balanced Loss Based on Effective Number of Samples*), combining effective sample weights with focal loss.

---

## Module Structure

```text
losses/
├── __init__.py                # Public API exports
├── loss_names.py              # LossName enumeration
├── factory.py                 # build_loss factory function
├── registry.py                # LOSS_REGISTRY mapping
├── cross_entropy.py           # Cross-entropy loss builders
├── weighted_cross_entropy.py  # Weighted cross-entropy builders
├── focal.py                   # Focal loss implementation
├── class_balanced_focal.py    # Class-balanced focal loss implementation
├── exceptions.py              # LossError hierarchy
└── README.md                  # Technical documentation
```

---

## Usage Example

```python
from losses import build_loss

# Build loss function from config
criterion = build_loss(config.loss)

# Compute loss
loss = criterion(logits, targets)
```

---

## Design Principles

- **Registry Pattern**: Losses registered under type-checked `LossName` enum values.
- **Fail-Fast Initialization**: Hyperparameter validation occurs during factory construction (`build_loss`).
- **Class Imbalance Focus**: Built-in support for medical imaging class imbalance via Focal and Class-Balanced loss variants.
