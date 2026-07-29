# Training Orchestration Module

## Overview

The `training` module is responsible for orchestrating component instantiation and execution for model training.

Following the **Factory Pattern**, this module builds neural network models, loss functions, optimizers, and learning rate schedulers from configuration files (`ProjectConfig`), keeping the training loop completely decoupled from concrete PyTorch class instantiations.

---

## Module Structure

```text
training/
├── __init__.py           # Public API exports
├── train.py              # Main CLI entry point for training
├── loss_factory.py       # Loss function factory
├── optimizer_factory.py  # Optimizer builder factory
├── scheduler_factory.py  # Learning rate scheduler factory
├── model_factory.py      # Neural network model factory
├── types.py              # Training type definitions
├── exceptions.py         # Factory exception hierarchy
├── utils.py              # Training helper utilities
└── README.md             # Technical documentation
```

---

## Component Factories

### 1. `ModelFactory`
Builds neural network classification architectures via TIMM (PyTorch Image Models).
- Supported backbones: `efficientnet_b0`, `resnet50`, `convnext_tiny`, `vit_base_patch16_224`.
- Replaces classification head to match target class count ($C=5$ for DDR dataset).

### 2. `LossFactory`
Instantiates configured loss functions:
- Cross-Entropy Loss (`cross_entropy`)
- Weighted Cross-Entropy (`weighted_cross_entropy`)
- Focal Loss (`focal`)
- Class-Balanced Focal Loss (`class_balanced_focal`)

### 3. `OptimizerFactory`
Constructs PyTorch optimizers:
- AdamW (`adamw`)
- Adam (`adam`)
- SGD (`sgd`)

### 4. `SchedulerFactory`
Constructs learning rate schedulers:
- Cosine Annealing (`cosine`)
- Step LR (`step`)
- OneCycle LR (`onecycle`)

---

## Usage Example

```python
from common.config import ConfigLoader
from training.model_factory import ModelFactory
from training.loss_factory import LossFactory
from training.optimizer_factory import OptimizerFactory
from training.scheduler_factory import SchedulerFactory

# Load configuration
config = ConfigLoader.load("configs/config.yaml")

# Instantiate components via factories
model = ModelFactory.build(config.model)
criterion = LossFactory.build(config.loss)
optimizer = OptimizerFactory.build(config.optimizer, parameters=model.parameters())
scheduler = SchedulerFactory.build(config.scheduler, optimizer=optimizer)
```

---

## Design Principles

- **Single Responsibility Principle**: Each factory constructs exactly one entity.
- **Configuration Driven**: All component hyperparameter specifications reside in YAML files.
- **Decoupled Architecture**: Swapping models, optimizers, or loss functions requires zero changes to training loop code.
