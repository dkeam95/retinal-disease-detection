# 🚀 Retinal Disease Detection — Execution & Evaluation Scripts

This directory contains all operational entrypoints, evaluation benchmarks, XAI audit suites, lesion segmenters, and clinical report generators for the Retinal Disease Detection system.

---

## 📁 Directory Structure

```
scripts/
├── training/
│   └── train_detection.py                  # CLI Faster R-CNN Lesion Detector Training
├── visualization/
│   ├── render_segmentation_slide_table.py  # 5-Column 3-Step Segmentation Grid Renderer (Slide 3)
│   ├── render_xai_slide_table.py           # 5-Column SOTA XAI Visual Explanation Grid Renderer (Slide 4)
│   └── render_architecture_diagram.py      # Architectural Graph Renderer (300 DPI)
├── evaluation/
│   ├── benchmark_xai.py                    # Side-by-Side Benchmark of 5 SOTA XAI Algorithms
│   ├── compare_detection_rounds.py         # 3-Round Resolution Benchmark (640 vs 1024 vs 1536 MAX)
│   ├── compare_lesion_ground_truth.py      # Validation against Official DDR Doctor TIF Masks
│   ├── test_detection_pipeline.py          # Single-run Lesion Detector Visualizer
│   ├── test_lesion_segmenter.py            # Algorithmic Lesion Marker Tester
│   └── test_severe_cases.py                # Grade 3 (Severe) & Grade 4 (Proliferative) Audit
└── reporting/
    ├── generate_master_clinical_reports.py # Executive Clinical & AI Analytics Report Generator
    ├── export_pdf_reports.py               # English PDF Executive Reports Compiler (4 PDFs)
    ├── generate_pdf_report.py              # Publication-Grade Executive PDF Report Generator
    └── run_demo_pipeline.py                # Full End-to-End Live Pipeline Demonstration
```

---

## 🛠️ Usage Instructions

All scripts can be executed directly from the project root directory using Python:

### 1. Training
- **Train Faster R-CNN Lesion Detector**:
  ```bash
  python scripts/training/train_detection.py --config configs/detection/exp_faster_rcnn_lesions_v3_1536.yaml
  ```

### 2. Evaluation & Benchmarks
- **Run 3-Round Detection Resolution Benchmark (640x640 vs 1024x1024 vs 1536x1536)**:
  ```bash
  python scripts/evaluation/compare_detection_rounds.py
  ```
- **Benchmark 5 SOTA XAI Algorithms (Grad-CAM, Grad-CAM++, Layer-CAM, Score-CAM, Integrated Gradients)**:
  ```bash
  python scripts/evaluation/benchmark_xai.py
  ```
- **Validate against Official Expert Doctor Ground-Truth TIF Masks**:
  ```bash
  python scripts/evaluation/compare_lesion_ground_truth.py
  ```
- **Evaluate High-Severity Cases (Severe & Proliferative DR)**:
  ```bash
  python scripts/evaluation/test_severe_cases.py
  ```

### 3. Executive Reporting
- **Generate Master Executive Clinical HTML Diagnostic Report**:
  ```bash
  python scripts/reporting/generate_master_clinical_reports.py
  ```
- **Generate Executive Publication-Grade PDF Report**:
  ```bash
  python scripts/reporting/generate_pdf_report.py
  ```
- **Run Complete Live Pipeline Demo**:
  ```bash
  python scripts/reporting/run_demo_pipeline.py
  ```

---

## 📊 Output Artifacts

- **HTML Reports**: `reports/master_clinical_reports/master_clinical_diagnostic_report.html`
- **PDF Report**: `reports/retinal_disease_detection_final_report.pdf`
- **Detection Benchmark**: `reports/figures/detection_comparison/detection_3round_benchmark.html`
- **XAI Benchmark**: `reports/figures/xai_benchmark/xai_benchmark_report.html`
