# Retinal Disease Detection — Deep Learning & SOTA XAI Suite

> A production-grade Deep Learning framework for automated 5-stage Diabetic Retinopathy (DR) classification from fundus images, featuring Multi-Model Ensembling, Error Analysis, and a SOTA 5-Algorithm Explainable AI (XAI) Suite with CLAHE Spotlight visualization.

---

## 📌 Overview

**Retinal Disease Detection** is a modular, reproducible Computer Vision framework built for automated Diabetic Retinopathy grading using fundus photography. The framework adheres to strict software engineering standards, including pipeline-driven modular architecture, static type checking, automated unit testing, configuration-driven experiment management, and advanced clinical explainability.

Trained and evaluated on the **DDR (Diabetic Retinopathy)** dataset supporting 5 ordinal severity grades.

### 📚 Research & Dataset References

- **Dataset**: [Hugging Face — DDR Dataset (`addone5/DDR-dataset`)](https://huggingface.co/datasets/addone5/DDR-dataset)
- **Publication**: [DDR: A dataset for diabetic retinopathy grading and lesion segmentation (ScienceDirect)](https://www.sciencedirect.com/science/article/pii/S2666389922001040)

---

## 🎯 Key Features & Module Capabilities

- **Pipeline-First Architecture**: Decoupled modules (`common`, `dataset`, `dataloader`, `model`, `losses`, `metrics`, `trainer`, `evaluation`, `inference`, `explainability`).
- **Multi-Model Weighted Ensemble**: Combines predictions from top-tier vision architectures (`ConvNeXt-Tiny v2` + `Swin-T v2` + `DenseNet-121 v2`) via weighted probability averaging.
- **SOTA 5-Algorithm Explainability Suite (`src/explainability`)**:
  1. **Baseline Grad-CAM**: Standard coarse spatial activation mapping.
  2. **Grad-CAM++**: Higher-order (2nd/3rd derivative) gradient weighting for multiple scattered lesion detection.
  3. **Layer-CAM**: Element-wise gradient spatial preservation for high-resolution vessel & lesion boundaries.
  4. **Score-CAM**: Gradient-free perturbation confidence scoring that completely eliminates gradient noise.
  5. **Integrated Gradients**: Axiomatic path-integral feature attribution for pixel-level micro-lesion pinpointing.
- **CLAHE Spotlight 4-Panel Visualization**: Replaces blurry heatmap overlays with CLAHE local contrast enhancement, color-coded contour spotlights by DR grade, and 2x2 multi-panel diagnostic reports.
- **Medical Metrics & Loss Functions**: Primary evaluation uses **Quadratic Weighted Kappa (QWK)** alongside Class-Balanced Focal Loss, Focal Loss, Macro-F1, Precision, Recall, and Confusion Matrix analysis.
- **Robust Training & Checkpointing Engine**: Automatic Mixed Precision (AMP), gradient accumulation, cosine learning rate scheduling, early stopping, and stateful checkpoint recovery.

---

## 🏆 Side-by-Side Benchmark Leaderboard (Round 1 vs Round 2)

| # | Architecture | Round 1 Test Loss | Round 2 Test Loss | Round 1 Test QWK | Round 2 Test QWK | Test Loss Improvement |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| 🥇 **1** | **ConvNeXt-Tiny** | 1.9002 | **0.3418** | 0.7499 | **0.7569** 🚀 | **-82.0%** (Project Champion!) |
| 🥈 **2** | **DenseNet-121** | 1.8149 | **0.3007** | 0.6825 | **0.6977** | **-83.4%** |
| 🥉 **3** | **EfficientNet-B0** | 2.0926 | **0.2164** | 0.6550 | **0.6972** | **-89.7%** |
| **4** | **ResNet-50** | 2.1145 | **0.3576** | 0.6802 | **0.6969** | **-83.1%** |
| **5** | **MobileNetV3-Large** | 2.8933 | **0.3309** | 0.6358 | **0.6767** | **-88.6%** |
| **6** | **Swin-T** | 1.9181 | **0.2902** | **0.7542** | 0.7256 | **-84.9%** |
| **7** | **ViT-Small** | 3.4229 | **0.5938** | 0.6059 | 0.5586 | **-82.7%** |

---

## 🩺 Diabetic Retinopathy Disease Severity Scale

The system evaluates fundus images according to the International Clinical Diabetic Retinopathy Disease Severity Scale (5 classes):

| Grade | Disease State | Description |
| :---: | :--- | :--- |
| **0** | No DR | Normal fundus image without detectable lesions. |
| **1** | Mild NPDR | Microaneurysms only. |
| **2** | Moderate NPDR | More than microaneurysms but less than severe NPDR. |
| **3** | Severe NPDR | Intraretinal hemorrhages, venous beading, or prominent IRMA. |
| **4** | Proliferative DR (PDR) | Neovascularization, vitreous/preretinal hemorrhage. |

---

## 📂 Repository Structure

```text
retinal-disease-detection/
├── configs/              # YAML experiment configurations
│   └── experiments/      # Individual model experiment configs
├── data/                 # Dataset directory (raw/ and processed/)
├── src/                  # Production source code
│   ├── checkpoint/       # Checkpoint state management & saving
│   ├── common/           # Core config loader, dataclasses, logging
│   ├── dataloader/       # PyTorch DataLoader factory & weighted samplers
│   ├── dataset/          # RetinalDataset interface & text annotation parser
│   ├── evaluation/       # ErrorAnalyzer, confusion matrix & evaluation loops
│   ├── experiments/      # Experiment comparator & leaderboard generator
│   ├── explainability/   # SOTA XAI Suite (Grad-CAM, Grad-CAM++, Layer-CAM, Score-CAM, IG, Spotlight)
│   ├── inference/        # RetinalPredictor & EnsemblePredictor engines
│   ├── losses/           # Class-Balanced Focal Loss, Focal Loss, Cross-Entropy
│   ├── metrics/          # QWK, Macro-F1, Accuracy, Precision, Recall
│   ├── model/            # Model architecture factory & timm backbones
│   ├── preprocessing/    # Augmentation pipelines (CLAHE, Albumentations)
│   ├── trainer/          # Training loop execution & step engine
│   └── training/         # High-level training orchestration
├── tests/                # Automated pytest unit & integration test suite (166 tests)
├── benchmark_xai.py      # XAI 5-algorithm side-by-side benchmark runner
├── run_demo_pipeline.py  # Complete live demonstration pipeline
├── pyproject.toml        # Build configuration, linters (ruff, mypy), dependencies
└── README.md             # Project documentation entry point
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

### 2. Live Demo Pipeline

Execute the live demonstration pipeline (Ensemble inference + Error analysis + CLAHE Spotlight visual audit):

```bash
python run_demo_pipeline.py
```

### 3. Running the SOTA XAI Benchmark

Evaluate and compare all 5 XAI algorithms side-by-side:

```bash
python benchmark_xai.py
```

Generates an interactive HTML comparison report at `reports/figures/xai_benchmark/xai_benchmark_report.html`.

### 4. Running Code Quality & Tests

```bash
# Run test suite
pytest

# Run static code analysis
ruff check .

# Run type checker
mypy src
```

---

## 🛠 Technology Stack

- **Core**: Python 3.11, PyTorch 2.7, Torchvision 0.22
- **Model Backbones**: `timm` (ConvNeXt, Swin Transformer, DenseNet, ResNet, EfficientNet)
- **Explainable AI**: Grad-CAM, Grad-CAM++, Layer-CAM, Score-CAM, Integrated Gradients, CLAHE
- **Computer Vision**: OpenCV, Albumentations, PIL
- **Configuration**: PyYAML, Dataclasses
- **Quality Assurance**: `pytest`, `pytest-cov`, `mypy`, `ruff`

---

## 📄 License

This project is released under the [MIT License](LICENSE).
