# System Specification: Retinal Disease Detection

## 1. Functional Specification

### 1.1 Purpose
The Retinal Disease Detection system provides an automated, reproducible machine learning pipeline for multi-class Diabetic Retinopathy (DR) classification using color fundus photography from the DDR dataset.

### 1.2 Input Data Specifications
- **Input Image Format**: JPEG / PNG color fundus images.
- **Input Resolution**: Configurable (Default: $512 \times 512$ pixels, 3 RGB channels).
- **Annotation Files**: Text files (`train.txt`, `valid.txt`, `test.txt`) containing space-separated image filenames and integer labels.

### 1.3 Target Output Specifications
- **Classification Output**: 5-class DR severity grade ($C \in \{0, 1, 2, 3, 4\}$):
  - `0`: No DR (Normal)
  - `1`: Mild NPDR
  - `2`: Moderate NPDR
  - `3`: Severe NPDR
  - `4`: Proliferative DR (PDR)
- **Model Predictions**: Softmax class probabilities and raw logits ($N \times 5$).

---

## 2. Technical Specification

### 2.1 Performance & Evaluation Metrics
- **Primary Clinical Metric**: **Quadratic Weighted Kappa (QWK)** ($\ge 0.85$ target performance).
- **Secondary Metrics**: Accuracy, Macro-F1 Score, Macro-Precision, Macro-Recall, Confusion Matrix ($5 \times 5$).

### 2.2 System Modules & API Interfaces

| Module | Location | Responsibilities | Key API Calls |
| :--- | :--- | :--- | :--- |
| **Config** | `src/common/config/` | Schema validation, YAML parsing into frozen dataclasses | `ConfigLoader.load(path)` |
| **Dataset** | `src/dataset/` | Image disk loading, annotation parsing | `RetinalDataset(config)` |
| **DataLoader**| `src/dataloader/` | Batching, worker allocation, class-weighted sampling | `DataLoaderFactory.create(...)` |
| **Preprocessing**| `src/preprocessing/` | Resizing, Albumentations augmentations, ImageNet norm | `build_train_pipeline(config)` |
| **Model** | `src/model/` | TIMM backbone feature extractor + linear head | `create_model(config)` |
| **Losses** | `src/losses/` | Cross-Entropy, Focal Loss, Class-Balanced Focal | `build_loss(config)` |
| **Metrics** | `src/metrics/` | QWK, F1, Accuracy, Confusion Matrix calculations | `build_metrics(names)` |
| **Trainer** | `src/trainer/` | Train/val step execution, AMP mixed precision, loops | `Trainer.train()` |
| **Checkpoint**| `src/checkpoint/` | State serialization, best/latest model management | `CheckpointManager.save/load` |

### 2.3 Quality & Engineering Standards
- **Python Compatibility**: Python 3.11+
- **Static Analysis**: Zero errors under `ruff check` and `mypy --strict`.
- **Test Suite**: Automated `pytest` unit tests covering configuration, dataset parsing, metrics calculation, loss initialization, and model building.
