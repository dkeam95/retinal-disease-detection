# DataLoader Module (`src/dataloader/`)

## Overview

The `dataloader` module builds PyTorch `DataLoader` instances with configurable worker allocation, memory pinning, shuffling policies, and weighted class balancing to combat severe dataset imbalance.

## Key Features

- **Class-Weighted Sampling**: Computes sample-specific weights to feed into `WeightedRandomSampler`, ensuring underrepresented disease grades are trained on adequately.
- **Custom Collate Functions**: Implements collating functions that aggregate individual `DataSample` objects into a unified `BatchSample` dataclass.

## API & Interfaces

### Classes

#### `DataLoaderFactory`
Factory class that builds training, validation, and test dataloaders.
- **`build(dataset: Dataset, config: DataLoaderConfig, is_training: bool = False) -> DataLoader`**
  Builds and configures a standard `DataLoader` with correct batching, shuffling, and worker threads.

#### `BatchSample`
Dataclass holding batched model inputs:
- `image`: torch.Tensor of shape `(N, C, H, W)`
- `label`: torch.Tensor of shape `(N,)`

### Functions

- **`build_weighted_sampler(dataset: RetinalDataset) -> WeightedRandomSampler`**
  Calculates category frequencies and generates a sampler that balances class representation per batch.
- **`collate_datasamples(batch: list[DataSample]) -> BatchSample`**
  Collates list of individual dataset sample dictionaries/dataclasses into a batched `BatchSample`.
- **`build_train_dataloader(dataset, config) -> DataLoader`**
- **`build_validation_dataloader(dataset, config) -> DataLoader`**
- **`build_test_dataloader(dataset, config) -> DataLoader`**

### Exceptions

- `InvalidBatchSizeError`: Raised when the batch size parameter is negative or zero.
- `InvalidNumWorkersError`: Raised when workers count is invalid.
- `InvalidSamplerError`: Raised if sample weight calculations encounter mismatches.

## Dependencies

- `torch`: Data utilities (`DataLoader`, `WeightedRandomSampler`).
