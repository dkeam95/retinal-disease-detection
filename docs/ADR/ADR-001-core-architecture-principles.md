# ADR-001 — Core Architecture Principles

**Status:** Accepted  
**Date:** 2026-07-17  

---

## Context

The **Retinal Disease Detection System** is a long-term machine learning engineering project aimed at automated classification of retinal pathologies (specifically 5-stage Diabetic Retinopathy) from fundus photography.

Over the project lifecycle, the codebase is expected to handle:

- A large number of independent modules;
- Benchmarking across multiple neural network architectures;
- Systematic ML experiments;
- Continuous feature expansion;
- Component reusability for downstream medical imaging projects.

Experience shows that many academic and experimental ML projects suffer from technical debt over time due to:

- Monolithic structures;
- Tight component coupling;
- Circular imports;
- Lack of clear domain boundaries;
- Absence of standardized architectural guidelines.

To ensure long-term maintainability, scalability, and testability, fundamental architectural principles must be established prior to full system implementation.

---

## Decision

The project adopts the following core architectural principles:

### 1. Pipeline-First Architecture

The system is designed around an **ML Pipeline**, rather than around a specific model.

The Pipeline is the primary architectural backbone. Data flows through a strict, sequential pipeline:

```text
Dataset
  │
  ▼
Data Pipeline
  │
  ▼
Image Preprocessing Pipeline
  │
  ▼
Training Pipeline
  │
  ▼
Experiment Pipeline
  │
  ▼
Evaluation Pipeline
  │
  ▼
Inference Pipeline
```

The system architecture must remain agnostic to specific neural network architectures. Swapping model backbones must not require modifications to data loading, preprocessing, or evaluation modules.

---

### 2. Three-Level Architecture Hierarchy

The codebase is organized into a three-level structural hierarchy:

#### Level 1: Domain
High-level area of responsibility. Examples:
- `src/dataset/`
- `src/preprocessing/`
- `src/model/`
- `src/trainer/`
- `src/training/`
- `src/losses/`
- `src/metrics/`
- `src/common/`

#### Level 2: Module
Each domain contains decoupled functional modules (e.g., `src/trainer/` contains `step.py`, `state.py`, `trainer.py`).

#### Level 3: Files & Standardized Layout
Every module adheres to a unified internal file structure template:

```text
module/
├── __init__.py      # Public API exports
├── module.py        # Core implementation
├── types.py         # Data structures and domain types
├── exceptions.py    # Module-specific exceptions
├── README.md        # Technical documentation
└── tests/           # Dedicated test cases
```

---

### 3. Single Responsibility Principle (SRP)

Each module and component is responsible for exactly one well-defined domain task.

- **Compliant**: `Trainer`, `CheckpointManager`, `DatasetParser`, `ImageValidator`.
- **Non-Compliant**: A generic `utils.py` file containing dozens of unrelated helper functions.

---

### 4. Explicit Public API

Inter-module interaction occurs strictly through explicit public APIs declared in `__init__.py`.

Internal implementation details are private. Direct imports of internal submodules from external packages are prohibited.

- **Correct**: `from src.training.trainer import Trainer`
- **Incorrect**: `from src.training.trainer.internal_impl import Trainer`

---

### 5. Unidirectional Dependency Flow

Dependencies must flow strictly downward through the pipeline stages:

```text
Data ──► Preprocessing ──► Training ──► Evaluation ──► Inference
```

Reverse dependencies are forbidden. For example, `Training` may depend on `Preprocessing`, but `Preprocessing` must never depend on `Training`.

---

### 6. Low Coupling

Modules must operate independently with minimal knowledge of other modules' internal states. Changes within one module must not cascade into changes across unrelated modules.

---

### 7. High Cohesion

All elements within a single module must serve a single functional objective. If a module begins addressing multiple disparate concerns, it must be refactored into separate modules.

---

### 8. Experiment-Driven Development

Model development proceeds through isolated, reproducible experiments.

Modifying multiple independent variables simultaneously in a single experiment is forbidden. Each experiment alters exactly one variable (e.g., a new architecture, a different loss function, or a distinct augmentation strategy).

Every experiment automatically logs and persists:
- Configuration snapshot (`config.yaml`);
- Model weights (checkpoints);
- Evaluation metrics (QWK, F1, Loss curves);
- Training execution logs.

---

### 9. Documentation-First Approach

Technical specifications and documentation must precede or accompany implementation.

Every module requires:
- `README.md` documenting purpose, responsibilities, and API;
- `types.py` defining data contracts;
- Unit tests validating behavior;
- Updates to `PROJECT_CONTEXT.md`, `DESIGN_DOC.md`, and `CHANGELOG.md`.

Structural architecture changes require a new Architecture Decision Record (ADR).

---

### 10. Architecture Evolution

ADRs maintain a permanent, immutable record of architectural choices. Existing ADRs are never edited retroactively to change past decisions; instead, a new ADR is created superseding the previous decision.

---

## Consequences

### Positive

- High maintainability and project clarity;
- Seamless modular testability;
- Decoupled components enabling rapid model experimentation;
- Reusable pipeline modules for future computer vision projects.

### Negative

- Higher initial overhead for project setup and boilerplate;
- Strict discipline required to maintain documentation and design contracts.

---

## Alternatives Considered

1. **Monolithic ML Layout**: Rejected due to high risk of tight coupling and unmaintainable technical debt.
2. **Layer-Based Architecture**: Rejected due to poor domain isolation in machine learning workflows.
3. **Feature-Based Architecture**: Considered, but Pipeline-First Architecture better aligns with the ML lifecycle.

---

## Related Documents

- [SPECIFICATION.md](../SPECIFICATION.md)
- [DESIGN_DOC.md](../DESIGN_DOC.md)
- [PROJECT_CONTEXT.md](../PROJECT_CONTEXT.md)
- [ADR-002: YAML Configuration](ADR-002-use-YAML-as-primary-configuration-format.md)
