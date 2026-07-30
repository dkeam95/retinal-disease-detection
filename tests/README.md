# Test Suite Directory (`tests/`)

Unit, integration, and regression test suite for Retinal Disease Detection.

## 📌 Overview

The `tests/` directory contains 166+ automated tests written using `pytest`. The test suite covers configuration validation, dataset parsing, model building, loss functions, metrics calculation, training step iterations, inference, XAI algorithms, and error analysis.

---

## 🗂️ Directory Structure

```
tests/
├── checkpoint/       # Tests for checkpoint manager and serialization
├── common/           # Tests for configuration loader, schemas, and validators
├── dataloader/       # Tests for batching, samplers, and data loaders
├── dataset/          # Tests for dataset parsing and PyTorch dataset item retrieval
├── evaluation/       # Tests for Evaluator engine and ErrorAnalyzer
├── experiments/      # Tests for ExperimentRunner and ExperimentComparator
├── explainability/   # Tests for Grad-CAM, Layer-CAM, Score-CAM, LesionSegmenter, Spotlight
├── inference/        # Tests for Predictor and EnsemblePredictor
├── losses/           # Tests for CrossEntropy, Focal Loss, Class-Balanced Focal Loss
├── metrics/          # Tests for Accuracy, Precision, Recall, F1, and QWK
├── model/            # Tests for backbone factory, classifier head, and model builders
├── preprocessing/    # Tests for ROI extraction, fundus cropping, and transforms
├── trainer/          # Tests for Trainer state, step execution, and exception handling
├── training/         # Tests for loss factory, optimizer factory, and scheduler factory
└── visualization/    # Tests for diagnostic plotting functions and report generation
```

---

## 💻 Running Tests

Run the full test suite with coverage:

```bash
make test
# Or directly using pytest:
pytest
```
