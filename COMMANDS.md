# Project CLI & Scripts Commands Reference

This document serves as a comprehensive reference guide for running the primary workloads, utilities, and Quality Assurance (QA) tools in the **Retinal Disease Detection** codebase.

All commands must be executed from the repository root with the active python environment enabled.

---

## 1. Master CLI Orchestrator (`main.py`)

The root `main.py` script acts as a master CLI entrypoint. It supports various workflows through the `--mode` argument:

```bash
# 1. Execute the full clinical demo pipeline (ensemble prediction + error profiling)
python main.py --mode demo

# 2. Generate detailed patient-level clinical HTML diagnostic charts
python main.py --mode master-report

# 3. Execute the benchmarking suite for SOTA Explainable AI (XAI) algorithms
python main.py --mode benchmark-xai

# 4. Generate the lesion segmentation pipeline overview grid (Slide 3)
python main.py --mode benchmark-seg

# 5. Run comparisons across the three Faster R-CNN detection rounds
python main.py --mode benchmark-det

# 6. Train the Faster R-CNN lesion detector using a specified configuration file
python main.py --mode train-detection --config configs/detection/exp_faster_rcnn_lesions_v3_1280.yaml

# 7. Generate publication-ready clinical PDF report aggregates
python main.py --mode pdf-report
```

---

## 2. Core Classifier Pipelines (Training, Inference & Evaluation)

These commands run the primary PyTorch neural network workflows for the 5-class Diabetic Retinopathy severity grading model.

### 2.1 Model Training
Trains the classification model using the designated experiment configuration:
```bash
python -m src.trainer.train --config configs/experiments/exp_07_convnext_tiny_v4.yaml
```

### 2.2 Model Evaluation
Evaluates the trained model checkpoints against specified validation or test splits:
```bash
python scripts/evaluation/evaluate_cls.py
```
*(By default, this script loads `configs/experiments/exp_07_convnext_tiny_v4.yaml` and executes validation on `experiments/exp_07_convnext_tiny_v4/checkpoints/best_model.pt`, printing classification reports, confusion matrices, and logging results to HTML).*

### 2.3 Single-Image Inference
Performs Diabetic Retinopathy grading on a single custom fundus photograph:
```bash
# Path-based CLI prediction
python scripts/evaluation/test_cls.py --image data/raw/test/007-1811-100.jpg

# Interactive GUI-based prediction (Tkinter fallback, falls back to terminal image choice in headless environments)
python scripts/evaluation/test_cls.py
```

---

## 3. Utility Scripts (`scripts/`)

Decoupled scripts located in the `scripts/` directory support localized tasks:

### 3.1 Dataset Acquisition & Verification (`scripts/download_dataset.py`)
* **Download & Verify Dataset:** Pulls the DDR Dataset from Hugging Face (`addone5/DDR-dataset`) directly into `data/raw/` and verifies dataset split files (`train.txt`, `valid.txt`, `test.txt`):
  ```bash
  # Download and verify DDR dataset
  python scripts/download_dataset.py

  # Or via Master CLI orchestrator
  python main.py --mode download-dataset

  # Verify dataset layout without downloading
  python scripts/download_dataset.py --verify-only
  ```

### 3.2 Dataset Analysis & Diagnostics (`scripts/analysis/`)
* **XML Dataset Parser:** Parses raw Pascal VOC XML labels, analyzes box stats, resolutions, and output class constraints:
  ```bash
  python scripts/analysis/analyze_detection_dataset.py
  ```

### 3.2 Decision Boundary Tuning (`scripts/classification/`)
* **Threshold Optimizer:** Optimizes decision thresholds to maximize the Quadratic Weighted Kappa (QWK) score on validation splits:
  ```bash
  python scripts/classification/optimize_thresholds.py
  ```

### 3.3 Training Workflows (`scripts/training/`)
* **Direct Detection Training:** Directly trains the Faster R-CNN lesion model bypassing `main.py`:
  ```bash
  python scripts/training/train_detection.py --config configs/detection/exp_faster_rcnn_lesions_v3_1280.yaml
  ```

### 3.4 Visualizations & Figure Rendering (`scripts/visualization/`)

* **Single/Custom Image Mask Overlay Visualizer (`scripts/visualization/visualize_mask_overlay.py`):**
  Generates precision color-coded lesion mask overlays (Exudates, Hemorrhages, Microaneurysms, Soft Exudates) on custom fundus images:
  ```bash
  # Interactive terminal image selection menu
  python scripts/visualization/visualize_mask_overlay.py

  # Process specific image by filename or path
  python scripts/visualization/visualize_mask_overlay.py --image 007-3396-200.jpg

  # Native OS GUI file selection dialog (Tkinter)
  python scripts/visualization/visualize_mask_overlay.py --gui
  ```

* **5-Column Lesion Segmentation Pipeline Grid (`scripts/visualization/render_segmentation_slide_table.py`):**
  Renders the 3-step segmentation pipeline grid (Raw Mask ➔ Top-Hat Filter ➔ Guided Edge Refinement ➔ Color Overlay) matching Slide 3:
  ```bash
  # Interactive terminal image selection menu
  python scripts/visualization/render_segmentation_slide_table.py

  # Process specific image by filename or path
  python scripts/visualization/render_segmentation_slide_table.py --image 007-1774-100.jpg

  # Native OS GUI file selection dialog
  python scripts/visualization/render_segmentation_slide_table.py --gui

  # Or via Master CLI orchestrator
  python main.py --mode benchmark-seg --images 007-1774-100.jpg
  ```

* **YOLO Bounding Box Overlay Visualizer (`scripts/visualization/visualize_yolo_predictions.py`):**
  Visualizes YOLOv8 tiled lesion detection bounding box overlays:
  ```bash
  # Process specific image
  python scripts/visualization/visualize_yolo_predictions.py --image data/raw/valid/007-1774-100.jpg

  # Render random test set samples
  python scripts/visualization/visualize_yolo_predictions.py --num-samples 5
  ```

* **Network Graph Generator (`scripts/visualization/render_architecture_diagram.py`):**
  Renders the neural network architecture structure as a visual graph diagram:
  ```bash
  python scripts/visualization/render_architecture_diagram.py
  ```

* **SOTA XAI Grid Exporter (`scripts/visualization/render_xai_slide_table.py`):**
  Exports 5-column comparative grid tables of SOTA XAI heatmaps matching Slide 4:
  ```bash
  python scripts/visualization/render_xai_slide_table.py
  ```

### 3.5 Diagnostic Reporting (`scripts/reporting/`)
* **HTML Diagnoses Exporter:** Compiles diagnostic logs into HTML tables:
  ```bash
  python scripts/reporting/generate_master_clinical_reports.py
  ```
* **PDF Exporter:** Converts HTML outputs to clinical patient PDF records:
  ```bash
  python scripts/reporting/export_pdf_reports.py
  ```
* **Standalone PDF Compiler:** Compiles master reports to PDF directly:
  ```bash
  python scripts/reporting/generate_pdf_report.py
  ```
* **Clinical Demo Pipeline Executable:** Orchestrates evaluation sequences locally:
  ```bash
  python scripts/reporting/run_demo_pipeline.py
  ```

---

## 4. Code Quality & Testing (QA)

Run these checks locally to ensure clean coding style and syntax compliance before staging commits:

```bash
# 1. Run static style checks ( Ruff )
ruff check .

# 2. Automatically fix secure linter issues and organize imports
ruff check . --fix

# 3. Format files matching PEP8/Black configurations
ruff format .

# 4. Strict static type analysis ( Mypy )
mypy src

# 5. Run the complete unit test suite with coverage report ( Pytest )
pytest
```
