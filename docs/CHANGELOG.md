# Changelog

All notable changes to the **Retinal Disease Detection** project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.6.0] - 2026-07-28

### Added
- **Trainer Engine & Checkpoint Management**: Implemented stateful `Trainer` execution engine with support for Automatic Mixed Precision (AMP), gradient accumulation, early stopping, and state persistence (`CheckpointManager`).
- **Comprehensive Documentation Suite**: Completed root `README.md`, `SPECIFICATION.md`, `DESIGN_DOC.md`, `PROJECT_CONTEXT.md`, and translated `ADR-001` & `ADR-002` into standardized English.
- **DataLoader Module & README**: Added `src/dataloader/README.md` and verified `DataLoaderFactory` with class-weighted sampling.

## [0.5.0] - 2026-07-25

### Added
- **Model Architecture & TIMM Integration**: Modular backbone + head architecture (`ClassificationModel`), `ModelOutput` dataclass, and factory pattern for model instantiation (`create_model`).
- **Loss Functions Registry**: Implemented Cross-Entropy, Weighted Cross-Entropy, Focal Loss, and Class-Balanced Focal Loss registry.
- **Evaluation Metrics Module**: Implemented Quadratic Weighted Kappa (QWK), Macro-F1, Precision, Recall, Accuracy, and Confusion Matrix with shape validation utilities.

## [0.4.0] - 2026-07-22

### Added
- **Image Preprocessing Pipeline**: Integrated Albumentations transformation pipelines for training, validation, and testing splits (`build_train_pipeline`, `build_validation_pipeline`).

## [0.3.0] - 2026-07-20

### Added
- **Dataset Module & Parser**: Implemented annotation file parsing (`load_annotations`), `RetinalDataset` PyTorch dataset interface, and dataset exception hierarchy.

## [0.2.0] - 2026-07-18

### Added
- **Configuration System**: Implemented `ConfigLoader` parsing YAML configurations into frozen dataclasses (`ProjectConfig`, `DatasetConfig`, `ModelConfig`, `TrainingConfig`).

## [0.1.0] - 2026-07-17

### Added
- **Repository Setup**: Initial project infrastructure, dependency management, linter configuration (`ruff`), static type checking (`mypy`), and test setup (`pytest`).
- **Architecture Decision Records**: Established `ADR-001` (Core Architecture Principles) and `ADR-002` (YAML Configuration Format).
