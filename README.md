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

---

## 🩺 Dataset & Disease Classification

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
├── configs/              # Centralized YAML configuration files
│   └── config.yaml       # Primary experiment configuration
├── data/                 # Dataset directory
│   ├── raw/              # Raw fundus images and split annotation files
│   └── processed/        # Preprocessed images / cached artifacts
├── docs/                 # Architectural decision records (ADRs), specs, & design docs
│   ├── ADR/              # Architecture Decision Records
│   ├── DESIGN_DOC.md     # Module design documentation
│   ├── PROJECT_CONTEXT.md# Current status and context
│   └── SPECIFICATION.md  # Functional and technical specifications
├── src/                  # Production source code
│   ├── checkpoint/       # Checkpoint saving, loading, and state management
│   ├── common/           # Shared core utilities (ConfigLoader, types, logging)
│   ├── dataloader/       # PyTorch DataLoader factory & class-balanced samplers
│   ├── dataset/          # Dataset interface & annotation parsers
│   ├── losses/           # Loss registry (CrossEntropy, Focal, Class-Balanced)
│   ├── metrics/          # Evaluation metrics (QWK, F1, Accuracy, Confusion Matrix)
│   ├── model/            # Neural network architectures, backbones, & heads
│   ├── preprocessing/    # Image augmentation & normalization pipelines
│   ├── trainer/          # Core training step execution and loop engine
│   └── training/         # High-level training orchestration and factories
├── tests/                # Automated pytest unit and integration test suite
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

### 2. Dataset Setup

Download the DDR dataset from [Hugging Face](https://huggingface.co/datasets/addone5/DDR-dataset) and place the dataset in `data/raw/`:

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
python -m src.training.train --config configs/config.yaml
```

### 4. Running Code Quality & Tests

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

## 🛠 Technology Stack

- **Core**: Python 3.11, PyTorch, Torchvision
- **Model Backbones**: `timm` (PyTorch Image Models)
- **Computer Vision**: OpenCV, Albumentations
- **Configuration**: PyYAML, Dataclasses
- **Quality Assurance**: `pytest`, `mypy`, `ruff`

---

## 📄 License

This project is released under the [MIT License](LICENSE).
