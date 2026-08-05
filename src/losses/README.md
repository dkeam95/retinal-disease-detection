# Losses Module (`src/losses/`)

## Overview

The `losses` module registers and initializes loss functions designed for training deep learning models, with a strong focus on ordinal classification and handling severe class imbalance.

## Key Features

- **Class Imbalance Resolution**: Features Weighted Cross-Entropy, Focal Loss, and Class-Balanced Focal Loss to prioritize hard-to-classify samples and rare classes.
- **Ordinal Target Formatting**: Support for Ordinal regression losses (e.g., CORAL loss).
- **Extensible Registry**: A centralized registry linking loss enums to implementation builders.

## API & Interfaces

### Classes

#### `FocalLoss(nn.Module)`
Focuses training on hard samples by dynamically scaling loss using focusing parameter $\gamma$ and balance coefficient $\alpha$.
- **`forward(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor`**

#### `ClassBalancedFocalLoss(nn.Module)`
Calculates class balance weights based on the effective number of samples per class ($E_n = \frac{1 - \beta^n}{1 - \beta}$) and applies them to Focal Loss.
- **`forward(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor`**

#### `CoralLoss(nn.Module)`
Consistent Ordinal Regression Loss for predicting ordinal scales.
- **`forward(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor`**

### Functions

- **`build_loss(config: LossConfig, class_counts: Optional[np.ndarray] = None) -> nn.Module`**
  Factory function returning the configured loss function.
- **`coral_logits_to_probs(logits: torch.Tensor) -> torch.Tensor`**
  Converts CORAL output logits to class probabilities.

### Exceptions

- `UnknownLossError`: Raised if the requested loss name is not registered.
- `LossInitializationError`: Raised if parameters like class counts are missing during class-balanced loss setup.

## Dependencies

- `torch`: PyTorch math operations.
