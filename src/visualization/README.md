# Visualization Module (`src/visualization`)

Plotting utilities, diagnostic chart generation, and automated report generation for Retinal Disease Detection.

## 📌 Overview

The `src/visualization` module provides publication-quality plotting tools for exploratory data analysis (EDA), model training convergence tracking, metric visualization, and clinical report generation.

---

## 🗂️ Directory Structure

```
src/visualization/
├── __init__.py           # Package exports (plots, LesionMaskVisualizer, ReportGenerator)
├── mask_overlay.py       # LesionMaskVisualizer & 5-Column Presentation Grid Renderers (Slides 3 & 4)
├── plots.py              # Matplotlib & Seaborn diagnostic chart plotting functions
└── report_generator.py   # Automated PDF/HTML evaluation report compiler
```

---

## 🔑 Key Components

### 1. Presentation Grid Renderers & Mask Visualizer (`mask_overlay.py`)
Provides production mask visualization tools:
* `render_slide3_comparison_grid()`: Renders the 5-column 3-step segmentation pipeline grid table image matching Slide 3 (`reports/figures/segmentation_pipeline_comparison_table.png`).
* `render_xai_slide4_comparison_grid()`: Renders the 5-column SOTA XAI visual explanation grid table image matching Slide 4 (`reports/figures/xai_pipeline_comparison_table.png`).
* `draw_mask_overlay()`: Color-coded lesion mask overlays (`MA` Green, `HE` Red, `EX` Yellow, `SE` Cyan/Blue).

### 2. Diagnostic Plots (`plots.py`)
Provides modular plotting functions:
* `plot_confusion_matrix()`: Normalized confusion matrix heatmap with class labels.
* `plot_training_curves()`: Multi-panel plot for Train/Val Loss, Accuracy, and QWK across training epochs.
* `plot_class_distribution()`: Bar chart visualizing dataset class imbalance ratios.
* `plot_roc_curves()`: One-vs-Rest ROC-AUC curves for multi-class DR severity.

### 2. Report Generator (`report_generator.py`)
Compiles comprehensive visual reports:
* Consolidates dataset statistics, model architecture hyperparameters, test set evaluation metrics, confusion matrices, and Grad-CAM heatmaps into structured reports.

---

## 💻 Usage Example

```python
from src.visualization import plot_confusion_matrix, plot_training_curves

# Plot training history
plot_training_curves(
    history={"train_loss": [...], "val_loss": [...], "val_qwk": [...]},
    save_path="reports/figures/training_history.png",
)

# Plot confusion matrix
plot_confusion_matrix(
    cm=confusion_matrix_matrix,
    class_names=CLASS_NAMES,
    save_path="reports/figures/confusion_matrix.png",
)
```
