# Inference Module (`src/inference/`)

## Overview

The `inference` module exposes prediction interfaces for deployment and testing. It supports single-image classification, ensemble scoring (averaging logits across multiple model weights), and lesion-masked image processing.

## Key Features

- **Single & Batch Predictions**: Provides streamlined APIs that load configs, load checkpoints, and process input images in a single call.
- **Model Ensembles**: Averages logits or probabilities from multiple models to improve generalization and grading stability.
- **Lesion Detection Overlays**: Automatically generates visual diagnostics by drawing boundary contours over predicted locations.

## API & Interfaces

### Classes

#### `RetinalPredictor`
The classification inference engine.
- **`from_checkpoint(checkpoint_path: Path, config: ProjectConfig, device: torch.device) -> RetinalPredictor`**: Instantiates predictor with loaded weights.
- **`predict(image: Union[str, Path, np.ndarray]) -> PredictionResult`**: Preprocesses image, performs forward pass, and returns results.

#### `EnsemblePredictor`
Binds multiple model checkpoints for ensemble classification.
- **`predict(image: np.ndarray) -> PredictionResult`**: Compiles average prediction across models.

#### `MaskedLesionPredictor`
Generates 3-step candidate lesion masks (Raw ➔ Top-Hat ➔ Guided Filter Refinement) from bounding boxes.
- **`predict_masked(image_input: Union[str, Path, np.ndarray]) -> MaskedDetectionResult`**: Runs FOV crop, bounding box detection, Top-Hat morphological filtering, and Guided Filter edge sharpening.

#### `PredictionResult`
Immutable container of inference outputs:
- `class_id`: int
- `class_name`: str
- `probabilities`: np.ndarray

## Dependencies

- `torch`: PyTorch model evaluations (`model.eval()`).
- `opencv-python`: Image ingestion.
