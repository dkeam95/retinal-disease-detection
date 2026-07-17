# Configuration System

## Purpose

The Configuration System is responsible for loading, validating, and providing
strongly typed project configuration objects.

The module serves as the single entry point for reading project configuration
files and converting them into immutable Python objects.

---

## Responsibilities

- Load YAML configuration files.
- Validate the root configuration structure.
- Convert configuration dictionaries into typed dataclasses.
- Raise meaningful custom exceptions.

This module is **not responsible** for:

- training configuration validation;
- model initialization;
- dataset loading;
- runtime parameter checking.

---

## Module Structure

```text
config/

├── __init__.py
├── config.py
├── exceptions.py
├── types.py
└── README.md
```

---

## Public API

```python
from common.config import ConfigLoader

config = ConfigLoader.load("configs/train.yaml")
```

Returns:

```python
ProjectConfig
```

---

## Configuration Flow

```text
YAML file
    │
    ▼
ConfigLoader.load()
    │
    ▼
Read YAML
    │
    ▼
Validate structure
    │
    ▼
Convert to dataclasses
    │
    ▼
ProjectConfig
```

---

## Main Classes

### ConfigLoader

Responsible for loading and validating configuration files.

### ProjectConfig

Root configuration object containing all project settings.

---

## Exceptions

The module may raise:

- `ConfigFileNotFoundError`
- `ConfigurationParsingError`
- `InvalidConfigurationError`

---

## Testing

The module is covered by unit tests located in:

```text
tests/common/config/
```

Current coverage includes:

- valid configuration loading;
- missing configuration file;
- malformed YAML;
- invalid root object;
- missing required sections.

---

## Design Principles

- Single Responsibility Principle
- Immutable configuration objects
- Strong typing
- Minimal public API
- Explicit error handling
