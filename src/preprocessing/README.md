# Preprocessing Module (`src/preprocessing/`)

## Overview

The `preprocessing` module is responsible for loading, augmenting, normalizing, and filtering fundus images. It contains specialized algorithms for fundus FOV (Field-Of-View) masking, Graham's processing, optic disc suppression, and Guided Filter boundary refinement.

## Key Features

- **Albumentations Pipelines**: Builds optimized pipelines for training (with active flips, rotations, and contrast changes) and validation/testing (standard ImageNet normalization and resizing).
- **Graham's Preprocessing**: Enhances contrast and suppresses illumination artifacts typical of fundus cameras.
- **FOV Extractor**: Extracts the active circular retina area, masking out black backgrounds and borders.
- **Optic Disc Detector**: Locates the optic nerve head to prevent false positives in lesion segmentation.

## API & Interfaces

### Classes

#### `FundusFOVExtractor`
Finds the active circular field of view in fundus images.
- **`extract_fov_mask(image: np.ndarray) -> np.ndarray`**: Generates a binary mask of the retina area.
- **`apply_grahams_preprocessing(image: np.ndarray, sigma: float = 10.0) -> np.ndarray`**: Computes local contrast enhancement.
- **`process(image: np.ndarray) -> FOVResult`**: Crops to bounding box of FOV and applies enhancement.

#### `GuidedFilterMaskRefiner`
Refines predicted lesion segmentation mask boundaries using guiding RGB images.
- **`refine(image: np.ndarray, mask: np.ndarray) -> np.ndarray`**: Applies edge-preserving guided filter smoothing.
- **`apply_top_hat(raw_mask: np.ndarray, kernel_size: int = 15) -> np.ndarray`**: Applies morphological top-hat filtering for spot isolation.
- **`refine_pipeline_steps(image_rgb: np.ndarray, raw_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]`**: Executes all 3 steps of the slide segmentation pipeline (Raw ➔ Top-Hat ➔ Final Refined Mask).

#### `OpticDiscDetector`
Locates the optic nerve head disc to mask it out.
- **`detect(image: np.ndarray) -> dict`**: Returns location bounding box.

### Functions

- **`build_train_pipeline(config: PreprocessingConfig) -> albumentations.Compose`**
  Builds the Albumentations composition for training data augmentation.
- **`build_validation_pipeline(config: PreprocessingConfig) -> albumentations.Compose`**
  Builds the Albumentations composition for validation (resizing + normalization).
- **`build_test_pipeline(config: PreprocessingConfig) -> albumentations.Compose`**
  Builds the Albumentations composition for test inference.

### Exceptions

- `PipelineBuildError`: Raised if Albumentations fails to initialize a composition.
- `PipelineExecutionError`: Raised if an image transformation fails at runtime.

## Dependencies

- `albumentations`: Main augmentation engine.
- `opencv-python`: Image filtering, resizing, contour mapping.
