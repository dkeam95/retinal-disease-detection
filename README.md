# Retinal Disease Detection

> A production-grade Deep Learning framework for automated 5-stage Diabetic Retinopathy (DR) classification from fundus images.

---

## 📌 Overview

**Retinal Disease Detection** is a modular, reproducible Computer Vision project built for automated Diabetic Retinopathy grading using fundus photography. The framework adheres to strict software engineering standards, including pipeline-driven modular architecture, static type checking, automated unit testing, and configuration-driven experiment management.

This project is trained and evaluated on the **DDR (Diabetic Retinopathy)** dataset, supporting 5 ordinal severity grades of retinal disease.

### 📚 Research & Dataset References

- **Dataset**: [Hugging Face — DDR Dataset (`addone5/DDR-dataset`)](https://huggingface.co/datasets/addone5/DDR-dataset)
- **Publication**: [DDR: A dataset for diabetic retinopathy grading and lesion segmentation (ScienceDirect)](https://www.sciencedirect.com/science/article/pii/S2666389922001040)

---

## 🎯 Key Features

- **Pipeline-First Architecture**: Completely decoupled modules (Data $\rightarrow$ Preprocessing $\rightarrow$ Model $\rightarrow$ Training $\rightarrow$ Evaluation $\rightarrow$ Inference).
- **Strictly Typed Configuration**: Fail-fast YAML configuration loaded into immutable dataclasses (`ProjectConfig`).
- **Flexible Model Registry**: PyTorch backbone/classifier separation supporting modern vision backbones via `timm` (e.g., EfficientNet-B0/B3/B4, ConvNeXt, ViT).
- **Medical Evaluation Metrics**: Primary evaluation uses **Quadratic Weighted Kappa (QWK)**, the clinical gold standard for ordinal DR grading, alongside Macro-F1, Precision, Recall, Accuracy, and Confusion Matrix.
- **Advanced Loss Functions**: Includes Cross-Entropy, Weighted Cross-Entropy, Focal Loss, and Class-Balanced Focal Loss for handling severe class imbalance.
- **Robust Training Engine**: Automatic Mixed Precision (AMP), gradient accumulation, customizable learning rate scheduling, early stopping, and stateful checkpoint management.
- **Explainability (XAI)**: Visualizing model focus areas using Grad-CAM, Integrated Gradients, Layer-CAM, Score-CAM, and lesion segmentation filters.

---

## 🩺 Dataset & Disease Classification

The system evaluates fundus images according to the International Clinical Diabetic Retinopathy Disease Severity Scale (5 classes):

| Grade | Disease State          | Description                                                  |
| :---: | :--------------------- | :----------------------------------------------------------- |
| **0** | No DR                  | Normal fundus image without detectable lesions.              |
| **1** | Mild NPDR              | Microaneurysms only.                                         |
| **2** | Moderate NPDR          | More than microaneurysms but less than severe NPDR.          |
| **3** | Severe NPDR            | Intraretinal hemorrhages, venous beading, or prominent IRMA. |
| **4** | Proliferative DR (PDR) | Neovascularization, vitreous/preretinal hemorrhage.          |

---

## 🛠 Technology Stack

- **Core Framework**: Python 3.11+, PyTorch 2.x, Torchvision
- **Neural Network Backbones**: `timm` (PyTorch Image Models)
- **Image Processing & Augmentations**: OpenCV, Albumentations
- **Configuration & Utilities**: PyYAML, Dataclasses
- **Quality Assurance**: `pytest`, `mypy` (strict), `ruff`
- **Visualization & Reports**: Matplotlib, Seaborn, Jinja2 (HTML reports)

---

## 📂 Folder Structure

```text
retinal-disease-detection/
├── configs/              # Centralized YAML configuration files
│   ├── config.yaml       # Primary experiment configuration
│   └── experiments/      # Specific experiment profiles (e.g., exp_07_convnext_tiny_v4.yaml)
├── data/                 # Dataset directory (raw and processed data splits)
│   ├── raw/              # Raw fundus images and split annotation files
│   └── processed/        # Preprocessed images / cached artifacts
├── docs/                 # Architecture specifications and design docs
│   ├── ADR/              # Architecture Decision Records
│   ├── DESIGN_DOC.md     # Module design documentation
│   ├── PROJECT_CONTEXT.md# Current status and sprint milestones
│   └── SPECIFICATION.md  # Functional and technical specifications
├── notebooks/            # Jupyter/Python notebooks for analysis (e.g. run_eda.py)
├── reports/              # Generated HTML reports and analysis summaries
├── scripts/              # Command-line utility scripts for execution
├── src/                  # Production source code
│   ├── checkpoint/       # Checkpoint saving, loading, and state management
│   ├── common/           # Shared core utilities (ConfigLoader, logging, types)
│   ├── dataloader/       # PyTorch DataLoader factory & class-balanced samplers
│   ├── dataset/          # Dataset interface & annotation parsers
│   ├── detection/        # Lesion detection algorithms and XML parsers
│   ├── evaluation/       # Performance evaluation and confusion matrix plots
│   ├── explainability/   # Model explainability (Grad-CAM, Integrated Gradients)
│   ├── inference/        # Predictors, model ensembles, and CLI scripts
│   ├── losses/           # Loss registry (CrossEntropy, Focal, Class-Balanced)
│   ├── metrics/          # Evaluation metrics (QWK, F1, Accuracy, Confusion Matrix)
│   ├── model/            # Neural network architectures, backbones, & heads
│   ├── preprocessing/    # Image augmentation, FOV extraction, & norm pipelines
│   ├── trainer/          # Training loop orchestrator & trainer state
│   ├── training/         # High-level factories for optimization components
│   └── visualization/    # Overlay rendering, plots, and report generation
├── tests/                # Automated pytest unit and integration test suite
├── .env.example          # Template for environment variables
├── pyproject.toml        # Build configuration, linters (ruff, mypy), dependencies
├── requirements.txt      # Frozen dependencies
└── README.md             # Project documentation entry point
```

---

## ⚙️ Environment Variables (`.env`)

The project supports loading parameters via environment variables. To set them up:

1. Copy the template:
   ```bash
   cp .env.example .env
   ```
2. Configure your paths and options inside `.env`:

   ```env
   # Path to the dataset root folder
   DATA_ROOT=data/raw

   # Active CUDA device IDs (e.g., "0" for single GPU, "" for CPU)
   CUDA_VISIBLE_DEVICES=0

   # Python execution configuration
   PYTHONPATH=src
   PYTHONUNBUFFERED=1

   # Logging level (DEBUG, INFO, WARNING, ERROR)
   LOG_LEVEL=INFO
   ```

---

## ⚡ Quick Start

### 1. Prerequisites & Installation

Ensure Python 3.11+ is installed. Clone the repository and set up the environment:

```bash
# Clone the repository
git clone https://github.com/your-org/retinal-disease-detection.git
cd retinal-disease-detection

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Dataset Setup

Download the DDR dataset from [Hugging Face](https://huggingface.co/datasets/addone5/DDR-dataset) and place it in the `data/raw/` directory:

```text
data/raw/
├── train/                # Training fundus images (.jpg)
├── valid/                # Validation fundus images (.jpg)
├── test/                 # Test fundus images (.jpg)
├── train.txt             # Format: <image_name.jpg> <label_id>
├── valid.txt
└── test.txt
```

### 3. Running Training

Train the model using the primary configuration:

```bash
python -m src.trainer.train --config configs/experiments/exp_07_convnext_tiny_v4.yaml
```

### 4. Running Evaluation, Inference & Mask Overlay Visualizations

Evaluate classification model performance on the test set:

```bash
python scripts/evaluation/evaluate_cls.py
```

Perform Diabetic Retinopathy grading on a single custom fundus image:

```bash
python scripts/evaluation/test_cls.py --image data/raw/test/001-0001-000.jpg
```

Render precision lesion mask overlays (Exudates, Hemorrhages, Microaneurysms, Soft Exudates):

```bash
# Interactive terminal selection menu
python scripts/visualization/visualize_mask_overlay.py

# Specific image selection by filename/path
python scripts/visualization/visualize_mask_overlay.py --image 007-3396-200.jpg

# Native OS GUI file browser dialog
python scripts/visualization/visualize_mask_overlay.py --gui
```

Render 5-column 3-step segmentation pipeline grid (Slide 3):

```bash
# Process custom image
python scripts/visualization/render_segmentation_slide_table.py --image 007-1774-100.jpg

# Native OS GUI file browser dialog
python scripts/visualization/render_segmentation_slide_table.py --gui
```

### 5. Running Code Quality & Tests

Validate code quality and test coverage:

```bash
# Run static code analysis
ruff check .

# Run type checker
mypy src

# Run test suite
pytest
```

---

## 📄 License

This project is released under the [MIT License](LICENSE).
