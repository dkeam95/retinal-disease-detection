# DataLoader Module

## Purpose

The `dataloader` module provides factory functions to instantiate PyTorch `DataLoader` objects for training, validation, and testing splits.

It encapsulates class-balanced weighted sampling (`WeightedRandomSampler`), custom batch collation (`collate_datasamples`), and multi-worker execution settings.

---

## Responsibilities

- Constructing PyTorch `DataLoader` instances driven by `DataLoaderConfig`.
- Providing `collate_datasamples` to stack `DataSample` instances into `BatchSample` tensors ($N, C, H, W$).
- Supporting class-balanced weighted sampling via `build_weighted_sampler` for handling imbalanced DR severity grades.
- Supporting pinned memory, persistent workers, and configurable worker threads.

---

## Public API

```python
from dataloader import DataLoaderFactory, DataLoaderConfig

# Create training dataloader
train_loader = DataLoaderFactory.build(
    dataset=train_dataset,
    config=config,
    training=True,
)

# Create validation / test dataloader
val_loader = DataLoaderFactory.build(
    dataset=val_dataset,
    config=config,
    training=False,
)
```

---

## Module Structure

```text
dataloader/
├── __init__.py      # Public API exports
├── factory.py       # DataLoaderFactory & collate_datasamples implementation
├── sampler.py       # Class-balanced WeightedRandomSampler builder
├── types.py         # DataLoaderConfig, BatchSample dataclasses
├── exceptions.py     # DataLoaderError, InvalidSamplerError
└── README.md        # Technical documentation
```

---

## Data Flow

```text
RetinalDataset + DataLoaderConfig
                │
                ▼
      DataLoaderFactory.build()
                │
                ├──────────────────────────────┐
                ▼                              ▼
    [WeightedRandomSampler]          [Standard Shuffling]
                │                              │
                └──────────────┬───────────────┘
                               ▼
                       collate_datasamples
                               │
                               ▼
            BatchSample (image, label, paths)
```

---

## Design Principles

- **Single Responsibility Principle**: Handles only batching, sampling strategies, and dataloader creation.
- **Class Balance Support**: Automatic calculation of inverse class frequencies to build a `WeightedRandomSampler` when `weight_class_balance=True`.
- **Fail-Safe Collation**: Handles numpy array to PyTorch tensor conversion, channel order adjustment (HWC to CHW), and ImageNet normalization.
