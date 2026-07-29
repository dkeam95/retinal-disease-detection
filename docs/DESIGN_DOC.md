# Design Document: Retinal Disease Detection System

## 1. System Overview

The **Retinal Disease Detection System** is designed as a modular, reproducible, and scalable computer vision framework for multi-class Diabetic Retinopathy (DR) severity grading on fundus images (DDR Dataset).

The framework strictly enforces separation of concerns across data handling, image transformation, model assembly, loss calculation, evaluation, and training orchestration.

---

## 2. Component Design Specifications

### 2.1 Configuration System (`src/common/config/`)

- **Immutability:** All configuration structures use frozen dataclasses (`@dataclass(frozen=True, slots=True)`) to ensure zero side-effects during execution.
- **Fail-Fast Validation:** Validation occurs during parsing (`ConfigLoader.load()`). Invalid types, missing keys, or unreachable files trigger custom exceptions (`ConfigFileNotFoundError`, `InvalidConfigurationError`).

### 2.2 Dataset & DataLoader (`src/dataset/` & `src/dataloader/`)

- **Annotation Parser:** Reads tab/space-separated text files (`train.txt`, `valid.txt`, `test.txt`) and returns typed `AnnotationRecord` instances.
- **PyTorch Dataset:** `RetinalDataset` handles disk image loading via OpenCV (`cv2.imread`), applying color channel conversion (BGR $\rightarrow$ RGB) and pipeline transformations before producing structured `DataSample` outputs.
- **DataLoader Factory:** Instantiates PyTorch `DataLoader` objects with support for standard or class-balanced weighted samplers (`WeightedRandomSampler`) to address class imbalance.

### 2.3 Preprocessing & Augmentation (`src/preprocessing/`)

- **Albumentations Integration:** Supports configurable augmentations including resizing, horizontal/vertical flips, random rotations, brightness/contrast adjustments, and standard ImageNet normalization ($Mean=[0.485, 0.456, 0.406]$, $Std=[0.229, 0.224, 0.225]$).
- **Separation of Pipelines:** Distinct pipeline builders exist for training (`build_train_pipeline`), validation (`build_validation_pipeline`), and testing (`build_test_pipeline`).

### 2.4 Neural Network Architecture (`src/model/`)

- **Backbone/Head Separation:** Features are extracted using TIMM backbones instantiated with `num_classes=0` to decouple feature representation from classification.
- **Classifier Head:** Independent linear classifier with dropout layers.
- **Structured Output:** Forward passes return a `ModelOutput` dataclass containing both classification `logits` and intermediate feature representations.

### 2.5 Loss & Metric Registries (`src/losses/` & `src/metrics/`)

- **Registry Pattern:** Loss functions and metrics are registered via enumerations (`LossName`, `MetricName`) rather than loose strings, avoiding runtime typos and enabling autocomplete.
- **Clinical Metric Focus:** Primary model evaluation relies on **Quadratic Weighted Kappa (QWK)**, penalizing misclassifications proportionally to ordinal severity distance (e.g., misclassifying Grade 0 as Grade 4 incurs a much heavier penalty than Grade 0 as Grade 1).

### 2.6 Trainer Engine & Checkpointing (`src/trainer/` & `src/checkpoint/`)

- **State Tracking:** `TrainerState` tracks current epoch, step, loss history, and best metric evaluation score.
- **Fault-Tolerant Checkpoints:** `CheckpointManager` serializes model weights, optimizer states, scheduler states, and metadata to enable training resume operations without state drift.

---

## 3. Test Isolation & Quality Assurance

- All test suites reside outside production code in `tests/`.
- Linters (`ruff`) and static type checkers (`mypy`) are executed in strict mode prior to deployment.
