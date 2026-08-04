# Explainability (XAI) Module (`src/explainability`)

Explainable AI (XAI) algorithms, saliency heatmap generation, lesion segmentation, and clinical spotlighting for Retinal Disease Detection.

## 📌 Overview

The `src/explainability` module provides medical-grade model interpretability tools. It enables clinicians and researchers to visualize which retinal regions (e.g., microaneurysms, hard exudates, hemorrhages, cotton wool spots) influenced the model's classification decision.

---

## 🗂️ Directory Structure

```
src/explainability/
├── __init__.py               # Package exports (GradCAM, GradCAMPlusPlus, LayerCAM, etc.)
├── engine.py                 # Unified XAI execution engine interface
├── gradcam.py                # Gradient-weighted Class Activation Mapping (Grad-CAM)
├── gradcam_plus_plus.py      # Grad-CAM++ for multi-lesion localization
├── layer_cam.py              # Layer-CAM for multi-scale feature layer activation
├── score_cam.py              # Score-CAM (gradient-free activation mapping)
├── integrated_gradients.py   # Integrated Gradients axiomatic attribution
├── lesion_segmenter.py       # Automated lesion segmentation & contour extraction from heatmaps
├── spotlight.py              # Clinical spotlighting & pathology bounding box visualizer
├── batch_audit.py            # Large-scale XAI dataset auditing framework
└── generate_heatmaps.py      # CLI script for bulk heatmap generation
```

---

## 🔑 Supported Algorithms

| Algorithm | Type | Best Used For |
| :--- | :--- | :--- |
| **Grad-CAM** | Gradient-based | General feature localization in final Conv layer |
| **Grad-CAM++** | Gradient-based | Small lesion detection (microaneurysms, punctate hemorrhages) |
| **Layer-CAM** | Gradient-based | Fine-grained localization across intermediate layers |
| **Score-CAM** | Gradient-free | High-fidelity activation maps without gradient noise |
| **Integrated Gradients** | Axiomatic | Precise pixel-level attribution baseline comparison |

---

## 🔍 Additional Clinical XAI Features

### 1. Lesion Segmenter (`lesion_segmenter.py`)
Thresholds saliency maps using adaptive Otsu thresholding to segment candidate lesion regions and compute lesion area coverage ratio.

### 2. Clinical Spotlight (`spotlight.py`)
Overlays bounding boxes and focus spotlights around high-activation retinal pathology regions, formatted for ophthalmologist inspection.

---

## 💻 Usage Example

```python
from src.explainability import GradCAM, LesionSegmenter, SpotlightVisualizer

# Initialize Grad-CAM for target layer
cam = GradCAM(model=model, target_layer=model.backbone.layer4[-1])

# Generate heatmap for target class
heatmap = cam.generate(input_tensor=img_tensor, target_class=3)  # Severe NPDR

# Segment lesion contours
segmenter = LesionSegmenter(threshold_ratio=0.6)
contours, mask = segmenter.segment(heatmap)

# Visualize clinical spotlight
spotlight = SpotlightVisualizer()
result_img = spotlight.render(original_image=img, heatmap=heatmap, contours=contours)
```
