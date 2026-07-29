# Configuration Module (`common/config`)

## Purpose

The `config` module provides strongly typed, immutable configuration loading for the Retinal Disease Detection system.

It parses YAML configuration files and maps their hierarchical key-value pairs into frozen Python dataclasses (`ProjectConfig`).

---

## Key Features

- **Immutable Dataclasses**: Configured via `@dataclass(frozen=True, slots=True)` to prevent unintentional runtime mutations.
- **Fail-Fast Schema Validation**: Instantly detects missing required sections, invalid data types, or file loading errors.
- **Unified Entrypoint**: `ConfigLoader.load(path)` serves as the single loader for the entire system.

---

## Hierarchical Configuration Tree

```text
ProjectConfig
├── DatasetConfig
├── PreprocessingConfig
├── ModelConfig
├── LossConfig
├── OptimizerConfig
├── SchedulerConfig
├── TrainingConfig
├── DataLoaderConfig
├── CheckpointConfig
├── EarlyStoppingConfig
├── MetricsConfig
└── ExperimentConfig
```

---

## Module Structure

```text
config/
├── __init__.py      # Public API exports
├── config.py        # ConfigLoader implementation
├── types.py         # ProjectConfig and child dataclass definitions
├── exceptions.py     # ConfigFileNotFoundError, InvalidConfigurationError
└── README.md        # Technical documentation
```

---

## Public API Usage

```python
from common.config import ConfigLoader

# Load configuration file
config = ConfigLoader.load("configs/config.yaml")

# Access strongly typed configuration attributes
print(config.model.architecture)  # "efficientnet_b0"
print(config.training.epochs)      # 100
```

---

## Design Principles

- **Declarative Separation**: Configuration data is completely separated from executable Python code.
- **Fail-Fast Error Handling**: Emits descriptive exceptions specifying the missing key or invalid field path.
