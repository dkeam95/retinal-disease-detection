# ADR-002 — Use YAML as Primary Configuration Format

**Status:** Accepted  
**Date:** 2026-07-17  

---

## Context

The machine learning pipeline requires a human-readable, flexible, and hierarchical file format to manage parameters spanning model architectures, preprocessing pipelines, training hyperparameters, loss functions, and dataset file paths.

---

## Alternatives Considered

- **JSON:** Lacks native support for inline comments, is structurally verbose, and presents manual editing challenges during complex experimentation setups.
- **Python-based Configs (`config.py`):** Introduces arbitrary code execution risks, violating declarative separation and configuration safety.
- **TOML:** Excellent for flat configurations, but less intuitive when defining deeply nested hierarchical structures common in computer vision experiments.

---

## Decision

Adopt **YAML** as the exclusive primary file format for all external configuration specifications.

Parsing is executed via the `PyYAML` library, followed by strict type-safe mapping into frozen Python dataclasses (`ProjectConfig`).

---

## Consequences

- Enhanced documentation capabilities within configuration files via inline comments.
- Enables fail-fast validation before model instantiation to shield the runtime environment from invalid types or missing parameters.
- Ensures total experiment reproducibility across runs.

---

## Related Documents

- [ADR-001: Core Architecture Principles](ADR-001-core-architecture-principles.md)
- [DESIGN_DOC.md](../DESIGN_DOC.md)
