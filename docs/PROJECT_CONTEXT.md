# Project Context: Retinal Disease Detection

## Current Project Status

- **Sprint 1 (Repository & Tooling Infrastructure):** Completed. Repository initialized, Python environment configured, linting (`ruff`), type checking (`mypy`), and testing framework (`pytest`) integrated.
- **Sprint 2 (Core Configuration):** Completed. Strongly-typed, fail-fast configuration loader (`ConfigLoader`) with frozen dataclasses implemented.
- **Sprint 3 (Dataset & DataLoader Module):** Completed. DDR dataset parsing (`load_annotations`), PyTorch `RetinalDataset`, weighted samplers, and `DataLoaderFactory` implemented.
- **Sprint 4 (Preprocessing & Augmentation):** Completed. Albumentations training, validation, and test transformation pipelines integrated.
- **Sprint 5 (Model & Loss Registry):** Completed. Modular TIMM backbone + classifier head architecture with `ModelOutput`, along with Cross-Entropy, Weighted Cross-Entropy, Focal Loss, and Class-Balanced Focal Loss registries.
- **Sprint 6 (Metrics & Trainer Engine):** Completed. QWK, F1, Accuracy, and Confusion Matrix metrics implemented; step execution, AMP mixed precision, stateful checkpointing, and `Trainer` engine verified with unit test suites.

---

## Technical Stack & Target Dataset

- **Primary Dataset**: [DDR (Diabetic Retinopathy) Dataset on Hugging Face](https://huggingface.co/datasets/addone5/DDR-dataset)
- **Target Task**: 5-class Diabetic Retinopathy severity grading (0: No DR, 1: Mild, 2: Moderate, 3: Severe, 4: Proliferative DR).
- **Core Library**: PyTorch 2.x, Torchvision, TIMM
- **Computer Vision**: OpenCV, Albumentations
- **Primary Clinical Metric**: Quadratic Weighted Kappa (QWK)

---

## Configuration Architecture

The configuration system uses a hierarchical root object `ProjectConfig`, encapsulating sub-configurations:

- `DatasetConfig` — Data root directory, annotation filenames, class counts.
- `PreprocessingConfig` — Pipeline steps, image dimensions, normalization values.
- `ModelConfig` — Model architecture, pretraining flag, class count.
- `LossConfig` — Loss function selection, label smoothing, focal gamma/alpha.
- `OptimizerConfig` — Optimizer type (AdamW, SGD, Adam), learning rate, weight decay.
- `SchedulerConfig` — Scheduler type (Cosine, Step, OneCycle), minimum learning rate.
- `TrainingConfig` — Epoch count, device allocation, determinism flags.
- `DataLoaderConfig` — Batch size, worker count, pin memory, shuffling.
- `CheckpointConfig` — Checkpoint directory, save strategies, monitoring metric.
- `EarlyStoppingConfig` — Early stopping patience, min delta, monitoring metric.
- `MetricsConfig` — Primary metrics (QWK, Accuracy), secondary metrics (F1).
- `ExperimentConfig` — Experiment name, random seed, output directory.

Data validation executes synchronously during `ConfigLoader.load()`.
