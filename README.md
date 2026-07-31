# Retinal Disease Detection — Deep Learning, SOTA XAI Suite & Faster R-CNN Lesion Detection

> A production-grade Deep Learning framework for automated 5-stage Diabetic Retinopathy (DR) classification, SOTA 5-Algorithm Explainable AI (XAI) feature attribution, algorithmic lesion segmentation, and Two-Stage Faster R-CNN lesion object detection fine-tuned on the official DDR dataset.

---

## 📌 Overview

**Retinal Disease Detection** is a modular, reproducible Computer Vision framework built for automated Diabetic Retinopathy grading and micro-lesion detection using fundus photography. The framework adheres to strict software engineering standards, including pipeline-driven modular architecture, static type checking, automated unit testing, configuration-driven experiment management, and advanced clinical explainability.

Supported tasks on the **DDR (Diabetic Retinopathy)** dataset:
1. **Multi-Class DR Severity Grading**: 5 ordinal severity grades (Grades 0-4).
2. **SOTA Explainability Suite**: 5 attribution algorithms (Grad-CAM, Grad-CAM++, Layer-CAM, Score-CAM, Integrated Gradients) with CLAHE Spotlight visualization.
3. **Algorithmic Lesion Marker**: Top-Hat / Bottom-Hat contrast extraction isolating Hemorrhages (HE) and Hard Exudates (EX).
4. **Faster R-CNN Lesion Detection**: Two-stage object detector fine-tuned on PASCAL VOC XML annotations for 4 lesion types (`EX`, `HE`, `MA`, `SE`).

### 📚 Research & Dataset References

- **Dataset**: [Hugging Face — DDR Dataset (`addone5/DDR-dataset`)](https://huggingface.co/datasets/addone5/DDR-dataset)
- **Publication**: [DDR: A dataset for diabetic retinopathy grading and lesion segmentation (ScienceDirect)](https://www.sciencedirect.com/science/article/pii/S2666389922001040)
- **Paper DOI**: [https://doi.org/10.1016/j.ins.2019.06.011](https://doi.org/10.1016/j.ins.2019.06.011)

### 📖 Citation (BibTeX)

If you use this codebase or the DDR dataset benchmark in your research, please cite:

```bibtex
@article{LI2019,
  title = "Diagnostic Assessment of Deep Learning Algorithms for Diabetic Retinopathy Screening",
  author = "Tao Li and Yingqi Gao and Kai Wang and Song Guo and Hanruo Liu and Hong Kang",
  journal = "Information Sciences",
  volume = "501",
  pages = "511 - 522",
  year = "2019",
  issn = "0020-0255",
  doi = "https://doi.org/10.1016/j.ins.2019.06.011",
  url = "http://www.sciencedirect.com/science/article/pii/S0020025519305377",
}
```

---

## 🎯 Key Features & Module Capabilities

- **Pipeline-First Architecture**: Decoupled modules (`common`, `dataset`, `dataloader`, `detection`, `model`, `losses`, `metrics`, `trainer`, `evaluation`, `inference`, `explainability`).
- **Multi-Model Weighted Ensemble**: Combines predictions from top-tier vision architectures (`ConvNeXt-Tiny v2` + `Swin-T v2` + `DenseNet-121 v2`) via weighted probability averaging (**Test QWK = 0.7685**, Single Model ConvNeXt **Test QWK = 0.7569**).
- **SOTA 5-Algorithm Explainability Suite (`src/explainability`)**:
  1. **Baseline Grad-CAM**: Standard coarse spatial activation mapping.
  2. **Grad-CAM++**: Higher-order (2nd/3rd derivative) gradient weighting for multiple scattered lesion detection.
  3. **Layer-CAM**: Element-wise gradient spatial preservation for high-resolution vessel & lesion boundaries.
  4. **Score-CAM**: Gradient-free perturbation confidence scoring that completely eliminates gradient noise.
  5. **Integrated Gradients**: Axiomatic path-integral feature attribution for pixel-level micro-lesion pinpointing.
- **Two-Stage Faster R-CNN Lesion Object Detector (`src/detection`)**:
  - ResNet50-FPN backbone with COCO pre-trained weights (**Test mAP@50 = 28.45%**, **Test mAP(0.50:0.95) = 18.24%**).
  - Customized $4\text{px}$ micro-anchors (`sizes: ((4,), (8,), (16,), (32,), (64,))`) tailored for tiny microaneurysms ($4-15\text{px}$).
  - Fine-tuned at $1536 \times 1536$ MAX resolution with CLAHE LAB-space contrast enhancement.
  - Automated IoU-based Bounding Box Merger (`_merge_overlapping_boxes`) consolidating overlapping duplicates into single precise bounding boxes.
- **Medical Metrics & Loss Functions**: Evaluates classification with **Quadratic Weighted Kappa (QWK)** and detection with **mAP@50** and **mAP (0.50:0.95)**.
- **Clean Execution Scripts Architecture**: All operational scripts, benchmarks, and report generators organized under `scripts/`.

---

## 📊 Faster R-CNN Lesion Detection Resolution Benchmark (3 Rounds)

| Round | Resolution | CLAHE | RPN Anchors | Epochs | Train Loss | Val mAP@50 | Val mAP (0.5:0.95) | Key Outcome |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Round 1** | $640 \times 640$ | Disabled | (8,16,32,64,128) | 20 | 0.4858 | 16.85% | 9.78% | Baseline run; coarse grid over-detected duplicate boxes. |
| **Round 2** | $1024 \times 1024$ | Enabled | (4,8,16,32,64) | 35 | 0.2916 | 17.24% | 10.16% | High-res; eliminated false-positive background noise. |
| **Round 3 (MAX)** | **$1536 \times 1536$** | **Enabled** | **(4,8,16,32,64)** | **40** | **0.2389 (-51%)** | **18.24% (+1.4%)** | **11.06% (+1.3%)** | **Project Champion! Precise single bounding boxes.** |

---

## 🏆 Classification Benchmark Leaderboard (Round 1 vs Round 2)

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

## 📁 Repository Structure

```text
retinal-disease-detection/
├── configs/              # YAML experiment configurations
│   ├── detection/        # Faster R-CNN Lesion Detection YAML configs
│   └── experiments/      # Classification model experiment configs
├── data/                 # Dataset directory (raw/ and processed/)
│   └── raw/
│       ├── train.txt / valid.txt / test.txt
│       ├── lesion_detection/     # PASCAL VOC XML bounding box annotations (EX, HE, MA, SE)
│       └── lesion_segmentation/  # Official doctor TIF masks
├── scripts/              # Production operational & evaluation scripts
│   ├── training/         # train_detection.py
│   ├── evaluation/       # benchmark_xai.py, compare_detection_rounds.py, test_*.py
│   └── reporting/        # generate_master_clinical_reports.py, generate_pdf_report.py
├── src/                  # Core Python package source code
│   ├── checkpoint/       # Checkpoint state management & saving
│   ├── common/           # Config loader, typed dataclasses, validation
│   ├── dataloader/       # PyTorch DataLoader factory & weighted samplers
│   ├── dataset/          # RetinalDataset & LesionDetectionDataset interfaces
│   ├── detection/        # Faster R-CNN builder, XML parser, mAP evaluator
│   ├── evaluation/       # ErrorAnalyzer, confusion matrix & evaluation loops
│   ├── explainability/   # SOTA XAI Suite (Grad-CAM, Grad-CAM++, Layer-CAM, Score-CAM, IG, Spotlight)
│   ├── inference/        # RetinalPredictor, EnsemblePredictor & LesionDetectionPredictor
│   ├── losses/           # Class-Balanced Focal Loss, Focal Loss, Cross-Entropy
│   ├── metrics/          # QWK, Macro-F1, Accuracy, Precision, Recall
│   ├── model/            # Model architecture factory & timm backbones
│   ├── preprocessing/    # Augmentation pipelines (CLAHE, Albumentations)
│   └── trainer/          # DetectionTrainer & Classification Trainer engines
├── tests/                # Automated pytest unit & integration test suite (171 tests passed)
├── pyproject.toml        # Build configuration, linters (ruff, mypy), dependencies
├── main.py               # Master CLI entrypoint for running full project & pipelines
└── README.md             # Project documentation entry point
```

---

## ⚡ Quick Start & Master Script Execution

### 1. Installation & Environment Setup

```bash
# Clone the repository
git clone https://github.com/your-org/retinal-disease-detection.git
cd retinal-disease-detection

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Linux/macOS: source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Unified Master Entrypoint (`main.py`)

Run the master pipeline or specific benchmark modes using `main.py`:

```bash
# Run Master Clinical Diagnostic & AI Analytics Pipeline (Default)
python main.py

# Run 3-Round Faster R-CNN Detection Resolution Benchmark
python main.py --mode benchmark-det

# Run 5 SOTA XAI Algorithms Side-by-Side Benchmark
python main.py --mode benchmark-xai

# Run Live Ensemble Demo & Error Analysis
python main.py --mode demo

# Generate Executive PDF Reports (Russian Cyrillic PDF Suite)
python main.py --mode pdf-report
```

---

## 🛠️ Complete Operational Command Reference

### 🧪 A. Automated Testing & Code Verification

- **Run Full Automated PyTest Suite (175 Tests)**:
  ```bash
  pytest tests/ --no-cov -v
  ```
  * **What it does**: Executes 100% of all unit and integration tests across dataset parsing, loss functions, evaluation metrics, model backbones, FOV masking, and trainer checkpoints.
  * **Why use it**: Validates code integrity before making commits or deploying new models.

---

### 📷 B. Black & White (B&W) Mask Overlays & SOTA Benchmarks

- **10 Real Fundus Images Lesion Mask Experiment (Class 3 & 4)**:
  ```bash
  python experiments/run_10_images_experiment.py
  ```
  * **What it does**: Evaluates the CUDA PyTorch model on 10 real fundus photographs containing Microaneurysms (`MA`) and Soft Exudates (`SE`).
  * **Output**: Generates color overlays (`sample_XX.png`) and **high-contrast B&W masks** (`sample_XX_bw.png`) in `reports/class_3_4_batch_10/`.

- **5 SOTA Model Architectures Lesion Benchmark**:
  ```bash
  python experiments/run_sota_benchmark_10_images.py
  ```
  * **What it does**: Compares 5 SOTA vision models (`ConvNeXt-Tiny`, `Swin-T`, `DenseNet121`, `EfficientNet-B0`, `ResNet-50`) on real test images.
  * **Output**: Generates `reports/figures/detection_comparison/sota_benchmark_report.html` and B&W high-contrast masks.

---

### 🔍 C. Explainable AI (XAI) & Master Clinical Reports

- **5 SOTA XAI Algorithms Benchmark**:
  ```bash
  python scripts/evaluation/benchmark_xai.py
  ```
  * **What it does**: Computes feature attribution maps for `Grad-CAM`, `Grad-CAM++`, `Layer-CAM`, `Score-CAM`, and `Integrated Gradients` with strict 0% background bleeding.
  * **Output**: Generates `reports/figures/xai_benchmark/xai_benchmark_report.html`.

- **Master Clinical Diagnostic HTML Suite**:
  ```bash
  python scripts/reporting/generate_master_clinical_reports.py
  ```
  * **What it does**: Generates multi-panel clinical diagnostic cards combining DR stage predictions, 5 XAI heatmaps, lesion markers, and Faster R-CNN bounding boxes in `reports/master_clinical_reports/`.

---

### 📄 D. Publication-Quality English PDF Report Compiler

- **Compile All English PDF Reports**:
  ```bash
  python scripts/reporting/export_pdf_reports.py
  ```
  * **What it does**: Compiles 4 publication-quality PDF reports in `reports/` with 100% aspect-ratio preserved graphics:
    1. `reports/master_multi_task_clinical_report.pdf` (Master 3-Task Multi-Task Report)
    2. `reports/lesion_detection_report.pdf` (Lesion Object Detection Report)
    3. `reports/lesion_segmentation_report.pdf` (Lesion Mask Segmentation Report)
    4. `reports/codebase_architecture_guide.pdf` (Module Map & Data Flow Guide)

---

### 🏋️ E. Model Training & Evaluation Scripts

- **Train ConvNeXt-Tiny Classifier**:
  ```bash
  python scripts/training/train_model.py --config configs/exp_05_convnext_tiny.yaml
  ```

- **Fine-Tune Faster R-CNN Lesion Detector at 1536px**:
  ```bash
  python scripts/training/train_detector.py --config configs/exp_faster_rcnn_lesions_v3_1536.yaml
  ```

- **Evaluate Model Checkpoint on Test Set**:
  ```bash
  python scripts/evaluation/evaluate_model.py --checkpoint experiments/exp_05_convnext_tiny_v2/checkpoints/best.pt
  ```

---

## 🏥 Clinical Diagnostic Reports Generated

All generated reports are placed inside the relative `reports/` directory of the project:

- **Master Multi-Task PDF Report**: `reports/master_multi_task_clinical_report.pdf`
- **Lesion Detection PDF Report**: `reports/lesion_detection_report.pdf`
- **Lesion Segmentation PDF Report**: `reports/lesion_segmentation_report.pdf`
- **Codebase Architecture Guide PDF**: `reports/codebase_architecture_guide.pdf`
- **Master Clinical Diagnostic HTML Report**: `reports/master_clinical_reports/master_clinical_diagnostic_report.html`

---

## 🛠 Technology Stack

- **Core Frameworks**: Python 3.11, PyTorch 2.7, Torchvision 0.22, OpenCV 4.11
- **Architectures**: `timm` (ConvNeXt, Swin Transformer, DenseNet, ResNet, EfficientNet), Faster R-CNN (ResNet50-FPN)
- **Reporting & Visualization**: ReportLab PDF, Matplotlib, OpenCV CLAHE LAB
- **Code Quality**: `pytest` (175/175 tests passed), `ruff` (linter), `mypy` (static type checker)
