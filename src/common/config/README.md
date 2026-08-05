# Config Module (`src/common/config/`)

## Overview

The `config` module provides a strongly-typed, immutable, and fail-fast configuration system. It loads parameters from YAML configuration files, parses them into nested Python dataclasses, and runs validation constraints to prevent runtime failures downstream.

## Key Features

- **Immutability**: All configuration schemas are defined using `@dataclass(frozen=True, slots=True)` to prevent accidental modifications at runtime.
- **Fail-Fast Validation**: The loader validates type compatibility, structure, directory existence, and numeric bounds immediately upon loading.
- **Strict Error Handling**: Throws clear, descriptive, custom exception types if parsing or validation fails.

## API & Interfaces

### Classes

#### `ConfigLoader`
Main entry point for loading configurations.
- **`load(config_path: Union[str, Path]) -> ProjectConfig`**
  Loads a YAML configuration file from the specified path, parses it, validates all parameters, and returns a frozen `ProjectConfig` hierarchy.

#### `ProjectConfig`
The root configuration object composed of sub-configs:
- `dataset`: `DatasetConfig` - Paths, files, and classes.
- `preprocessing`: `PreprocessingConfig` - Image size, normalizations, and augmentation probabilities.
- `dataloader`: `DataLoaderConfig` - Batch size, workers, and shuffling settings.
- `model`: `ModelConfig` - Architecture, dropout rate, and pretraining flags.
- `loss`: `LossConfig` - Choice of loss and focal scaling factors.
- `optimizer`: `OptimizerConfig` - Optimizer choice, learning rate, and weight decay.
- `scheduler`: `SchedulerConfig` - Learning rate scheduling settings.
- `training`: `TrainingConfig` - Device type, epochs, and determinism.
- `checkpoint`: `CheckpointConfig` - Path settings for saving state dicts.
- `early_stopping`: `EarlyStoppingConfig` - Stop patience and monitoring criteria.
- `mixed_precision`: `MixedPrecisionConfig` - AMP configuration.
- `metrics`: `MetricsConfig` - Primary and secondary metrics to calculate.
- `experiment`: `ExperimentConfig` - Seed and experiment naming metadata.

### Functions

- **`validate_config(config: ProjectConfig) -> None`**
  Validates parameter ranges, paths existence, and architectural compatibility. Raises `InvalidConfigurationError` if check fails.

### Exceptions

- `ConfigurationError`: Base class for configuration exceptions.
- `ConfigFileNotFoundError`: Raised when the YAML file is not found.
- `ConfigurationParsingError`: Raised when YAML parsing fails.
- `InvalidConfigurationError`: Raised when any field fails validation constraints.

## Dependencies

- `pyyaml`: YAML parsing.
- `typing_extensions` (if backporting slots/frozen dataclasses).
