# Retinal Disease Detection

> A production-grade Deep Learning framework for retinal disease classification using fundus images.

---

## 🏗 Overview

Retinal Disease Detection is a professional Computer Vision project focused on automatic classification of retinal diseases. The project is built following strict software engineering practices: modularity, full type-checking, automated testing, and reproducible experiments.

## 🚀 Project Status

**Current Stage**: ✅ Sprint 1 & 2 Completed

- **Sprint 1**: Repository Infrastructure & Tooling (Ruff, Mypy, Pytest).
- **Sprint 2**: Configuration System (Strictly typed, immutable YAML-based configs).

## 🛠 Technology Stack

- **Language**: Python 3.11
- **Deep Learning**: PyTorch, Torchvision
- **Computer Vision**: OpenCV, Albumentations
- **Configuration**: PyYAML
- **Quality Assurance**: pytest, mypy, ruff

## 📂 Repository Structure

```text
retinal-disease-detection/
├── configs/          # Experiment & system configuration files
├── data/             # Data storage (raw, processed, metadata)
├── docs/             # ADR, Design Documents, Context
├── src/              # Source code
│   └── common/
│       └── config/   # Configuration loading & validation logic
├── tests/            # Test suite
└── pyproject.toml    # Project dependencies and tool configurations
```
