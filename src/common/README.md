# Common Module (`src/common`)

Common utilities, canonical class definitions, and configuration system shared across the entire Retinal Disease Detection pipeline.

## 📌 Overview

The `src/common` module acts as the core foundational module of the project. It defines domain-level constants, medical class mappings, and the configuration management subsystem.

---

## 🗂️ Directory Structure

```
src/common/
├── __init__.py           # Package exports (DRClass, CLASS_NAMES, ConfigLoader, etc.)
├── classes.py            # Canonical Diabetic Retinopathy (DR) class definitions
└── config/               # Configuration management subsystem
    ├── README.md         # Detailed configuration subsystem documentation
    ├── exceptions.py     # Custom configuration exception hierarchy
    ├── loader.py         # ConfigLoader implementation with YAML parsing & deep-merge
    ├── types.py          # Frozen dataclass schemas (Config, DatasetConfig, ModelConfig, etc.)
    └── validator.py      # Runtime configuration validation logic
```

---

## 🔑 Key Components

### 1. Medical Class Definitions (`classes.py`)

Defines the 5-stage International Clinical Diabetic Retinopathy (ICDR) scale:

* **`DRClass(IntEnum)`**:
  * `0`: `NO_DR` — Healthy retina
  * `1`: `MILD_NPDR` — Mild Nonproliferative DR
  * `2`: `MODERATE_NPDR` — Moderate Nonproliferative DR
  * `3`: `SEVERE_NPDR` — Severe Nonproliferative DR
  * `4`: `PROLIFERATIVE_DR` — Proliferative DR (Advanced stage)
* **`CLASS_NAMES`**: Human-readable label tuple mapped index-by-index to `DRClass` values.

### 2. Configuration Subsystem (`config/`)

Manages typed, validated, and immutable YAML configurations:
* **Deep Merge**: Merges experiment-specific YAML overrides on top of `config/config.yaml`.
* **Validation**: Checks path existence, parameter ranges, and consistency before execution starts.

---

## 💻 Usage Example

```python
from src.common import CLASS_NAMES, ConfigLoader, DRClass

# Access class mappings
print(DRClass.NO_DR.value)  # Output: 0
print(CLASS_NAMES[DRClass.PROLIFERATIVE_DR])  # Output: "Proliferative DR"

# Load & validate configuration
config = ConfigLoader.load("configs/config.yaml")
print(config.model.name)
```
