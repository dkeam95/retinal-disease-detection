# Project Context: Retinal Disease Detection

## Current Project Status

- **Sprint 1 (Repository Infrastructure):** Completed. Repository initialized, virtual environment configured, static linters (`ruff`, `mypy`) and testing framework (`pytest`) successfully integrated.
- **Sprint 2 (Core Configuration):** Completed. A robust, strictly typed configuration core has been implemented and validated via automated test suites.

## Implemented Configuration Architecture

The configuration pipeline utilizes a hierarchical root structure headed by `ProjectConfig`, which encapsulates the following child immutable configurations:

- `DatasetConfig`
- `TrainingConfig`
- `ModelConfig`
- `ExperimentConfig`
- `PreprocessingConfig`

Data validation and object mapping execute synchronously during the execution of `ConfigLoader.load()`.
