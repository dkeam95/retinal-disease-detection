# Inference Module (`src/inference`)

Single-image, batch, and ensemble inference pipeline for Retinal Disease Detection models.

## 📌 Overview

The `src/inference` module provides high-performance inference utilities for deploying trained Diabetic Retinopathy models. It supports single-image prediction, batch processing, and multi-model ensembling (combining predictions from multiple architectures for higher clinical reliability).

---

## 🗂️ Directory Structure

```
src/inference/
├── __init__.py           # Package exports (Predictor, EnsemblePredictor)
├── predict.py            # CLI script for running inference from command line
├── predictor.py          # Core single/batch Predictor class
└── ensemble.py           # Multi-model EnsemblePredictor (probability averaging & voting)
```

---

## 🔑 Key Components

### 1. `Predictor` (`predictor.py`)
Encapsulates model loading, image preprocessing, forward pass execution, and post-processing:
* Returns predicted `DRClass`, human-readable class name, confidence score, and raw class probability vector.
* Supports GPU/CPU execution and batch inference.

### 2. `EnsemblePredictor` (`ensemble.py`)
Combines predictions from multiple distinct model architectures (e.g., ResNet50 + EfficientNet-B0 + DenseNet121):
* **Soft Voting (Probability Averaging)**: Averages soft probabilities across models before selecting max class.
* **Weighted Ensembling**: Assigns architecture weights based on validation QWK performance.

### 3. CLI Script (`predict.py`)
Run inference directly from the command line:
```bash
python -m src.inference.predict --image path/to/fundus.png --checkpoint models/checkpoints/best_model.ckpt
```

---

## 💻 Usage Example

```python
from src.inference import EnsemblePredictor, Predictor

# Single model inference
predictor = Predictor.from_checkpoint("models/checkpoints/best_model.ckpt", device="cuda")
result = predictor.predict("data/test/sample_01.png")

print(f"Predicted Class: {result.class_name}")
print(f"Confidence:      {result.confidence:.2%}")

# Multi-model Ensemble inference
ensemble = EnsemblePredictor.from_checkpoints(
    checkpoint_paths=[
        "models/checkpoints/resnet50.ckpt",
        "models/checkpoints/densenet121.ckpt",
        "models/checkpoints/efficientnet_b0.ckpt",
    ],
    device="cuda",
)
ensemble_result = ensemble.predict("data/test/sample_01.png")
print(f"Ensemble Prediction: {ensemble_result.class_name} ({ensemble_result.confidence:.2%})")
```
