# Data Directory (`data/`)

Data storage hierarchy for raw fundus photography, preprocessed datasets, and metadata splits.

## 📌 Overview

This directory adheres to standard data engineering practices to isolate raw source images from processed tensors and cropped regions of interest (ROI).

> ⚠️ **Note**: Raw dataset files and preprocessed image tensors are ignored by Git (via `.gitignore`). Only directory placeholders (`.gitkeep`) and CSV metadata splits are tracked.

---

## 🗂️ Directory Structure

```
data/
├── raw/         # Unmodified raw retinal fundus images (e.g., APTOS 2019 / EyePACS)
├── interim/     # Intermediate transformations (cropped FOV, Ben Graham preprocessed images)
├── processed/   # Final dataset splits (train.csv, val.csv, test.csv metadata)
└── external/    # External annotations or supplementary datasets
```

---

## 📋 CSV Metadata Format

Dataset CSV files in `data/processed/` must contain:

| Column | Type | Description |
| :--- | :--- | :--- |
| `id_code` | string | Unique image filename identifier |
| `diagnosis` | int (0-4) | Target Diabetic Retinopathy severity grade |
| `file_path` | string | Relative or absolute path to target image |
