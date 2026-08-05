# Visualization Module (`src/visualization/`)

## Overview

The `visualization` module converts model outputs and evaluation statistics into charts, comparison grids, and interactive reports. It produces side-by-side diagnostic summaries and HTML-based reports.

## Key Features

- **Lesion Mask Overlays**: Alpha-blended semi-transparent overlays and sharp contour strokes for 4 lesion types (EX, HE, MA, SE).
- **Segmentation Pipeline Comparison Grids**: 5-column breakdown matching Slide 3 (Raw Mask ➔ Top-Hat Filter ➔ Guided Edge Refinement ➔ Color Overlay).
- **Training Curve Plotting**: Automatically draws training/validation loss and QWK curves across epochs.
- **Confusion Matrix Rendering**: Generates normalized $5 \times 5$ confusion matrices using Seaborn heatmaps.
- **HTML Report Generation**: Compiles model architectures, parameters, metrics plots, and XAI overlay images into a standalone interactive HTML report.

## Executable Visualizer Scripts (`scripts/visualization/`)

- **`visualize_mask_overlay.py`**: Interactive CLI, file argument, or native GUI dialog for rendering mask overlays on any custom fundus image.
  ```bash
  python scripts/visualization/visualize_mask_overlay.py --image 007-3396-200.jpg
  python scripts/visualization/visualize_mask_overlay.py --gui
  ```
- **`render_segmentation_slide_table.py`**: Renders 5-column segmentation pipeline comparison grid (Slide 3) for custom images.
  ```bash
  python scripts/visualization/render_segmentation_slide_table.py --image 007-1774-100.jpg
  python scripts/visualization/render_segmentation_slide_table.py --gui
  ```

## API & Interfaces

### Classes

#### `LesionMaskVisualizer`
Renders annotated image grids highlighting pathology types.
- **`draw_mask_overlay(image_rgb: np.ndarray, masks: list|dict|np.ndarray, labels: list|None = None, draw_contours: bool = True, high_contrast_bw: bool = False) -> np.ndarray`**
- **`draw_side_by_side(image_rgb: np.ndarray, masks: list|dict|np.ndarray) -> np.ndarray`**
- **`render_slide3_comparison_grid(class_samples: dict, cell_size: tuple = (240, 240)) -> np.ndarray`**
  Renders a 5-column comparison grid showing raw images, raw masks, top-hat filtered outputs, boundary refined masks, and color overlays.

#### `HTMLReportGenerator`
Generates comprehensive clinical model summary documents.
- **`generate(report_data: dict, output_filepath: Path) -> None`**: Evaluates HTML layouts using Jinja2 templates.

### Functions

- **`plot_confusion_matrix(cm: np.ndarray, class_names: list[str]) -> plt.Figure`**
  Generates a Seaborn heatmap representing the confusion matrix.
- **`plot_learning_curves(history: dict) -> plt.Figure`**
  Plots learning curves for loss and accuracy/kappa.

## Dependencies

- `opencv-python`: High-resolution OpenCV image blending and contour strokes.
- `matplotlib`: Graph rendering.
- `seaborn`: Specialized heatmaps.
- `jinja2`: HTML templates compilation.
