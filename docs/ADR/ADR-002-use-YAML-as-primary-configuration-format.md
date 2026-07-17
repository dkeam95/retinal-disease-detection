# ADR-002: Use YAML as the Primary Configuration Format

## Status

Accepted

## Context

The machine learning pipeline requires a human-readable, flexible, and hierarchical file format to seamlessly manage multiple parameters spanning deep learning model architectures, preprocessing pipelines, training loops, and data paths.

## Alternatives Considered

- **JSON:** Lacks native support for inline commenting, structurally verbose, and presents manual editing challenges during complex experimentation setups.
- **Python-based Configs (config.py):** Introduces arbitrary code execution risks, violating declarative separation and configuration safety.
- **TOML:** Excellent for flat configurations, but less human-readable when defining deeply nested hierarchical dictionary structures common in advanced ML experiments.

## Decision

Adopt **YAML** as the exclusive primary file format for all external configuration specifications. Parsing is executed via the `PyYAML` library, followed by strict type-safe mapping into frozen Python dataclasses.

## Consequences

- Enhanced explicit documentation capabilities within configuration setups via inline code comments.
- Introduces a mandatory initial validation step to shield the runtime environment from missing parameters or typographical errors in configuration field keys.
