# Explainability Module (`src/explainability/`)

## Overview

The `explainability` module provides algorithms and tools for Explainable AI (XAI) in clinical decision support. It generates attribution maps and segmentation overlays highlighting retinal lesions (microaneurysms, hemorrhages, exudates) that guided classification predictions.

## Key Features

- **Pixel Attribution Maps**: Computes saliency maps using Grad-CAM++, Score-CAM, Layer-CAM, and Integrated Gradients.
- **Lesion Localization Filters**: Custom filters to extract and highlight diabetic retinopathy markers (HE, MA, EX, SE).
- **Spotlight Visualizer**: Annotates focus regions to improve diagnostic interpretability for clinical reviews.

## API & Interfaces

### Classes

#### `GradCAMPlusPlus`
Generates gradient-weighted class activation maps.
- **`generate(input_tensor: torch.Tensor, target_class: int) -> np.ndarray`**: Generates heatmaps.
- **`save_visualization(heatmap: np.ndarray, original_img: np.ndarray, filepath: Path)`**: Overlays and saves heatmap.

#### `IntegratedGradients`
Calculates pixel-level feature attribution relative to a neutral baseline image.
- **`generate(input_tensor: torch.Tensor, target_class: int, steps: int = 50) -> np.ndarray`**

#### `LesionSegmenter`
Extracts lesion-specific segments and renders clinical overlays.
- **`detect_exudates(image: np.ndarray) -> list[dict]`**: Returns contour and spatial metadata for bright hard exudates (EX).
- **`detect_hemorrhages(image: np.ndarray) -> list[dict]`**: Returns contour and spatial metadata for dark hemorrhages/microaneurysms (HE/MA).
- **`annotate(image_rgb: np.ndarray, draw_exudates: bool = True, draw_hemorrhages: bool = True, show_legend: bool = True) -> tuple[np.ndarray, dict]`**: Renders color-coded clinical circle annotations on top of the fundus image.

### Functions

- **`clahe_enhance(image: np.ndarray) -> np.ndarray`**
  Applies Contrast Limited Adaptive Histogram Equalization to highlight small lesions (e.g., microaneurysms).

## Dependencies

- `opencv-python`: Heatmap and image masking algorithms.
- `torch`: Backward gradients hook parsing.
