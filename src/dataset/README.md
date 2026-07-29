# Dataset Module

## Purpose

The `dataset` module is responsible for parsing annotation files, validating image records, reading color fundus images from disk, and serving structured samples via the PyTorch `Dataset` interface.

This module is designed specifically for retinal disease datasets such as the **DDR (Diabetic Retinopathy)** dataset.

---

## DDR Dataset & Disease Grades

The module supports 5 ordinal severity grades for Diabetic Retinopathy classification:

- **0 — No DR**: Normal fundus image.
- **1 — Mild NPDR**: Microaneurysms only.
- **2 — Moderate NPDR**: More than microaneurysms, less than severe NPDR.
- **3 — Severe NPDR**: Intraretinal hemorrhages, venous beading, IRMA.
- **4 — Proliferative DR (PDR)**: Neovascularization, vitreous hemorrhage.

Dataset Source: [Hugging Face — `addone5/DDR-dataset`](https://huggingface.co/datasets/addone5/DDR-dataset)

---

## Responsibilities

This module is **strictly responsible** for:

- Parsing annotation text files (`train.txt`, `valid.txt`, `test.txt`);
- Validating record paths and label ranges;
- Loading image files from disk using OpenCV (`cv2.imread`);
- Converting color space from BGR to RGB;
- Constructing immutable `DataSample` objects;
- Exposing a PyTorch-compatible `Dataset` (`RetinalDataset`).

This module does **not** perform:

- Image preprocessing or data augmentation (handled by `preprocessing`);
- Batching, worker allocation, or sampling (handled by `dataloader`);
- Model instantiation, loss calculation, or training loops.

---

## Public API

```python
from dataset import RetinalDataset, load_annotations

# Parse annotation records
records = load_annotations(
    annotation_file=Path("data/raw/train.txt"),
    image_directory=Path("data/raw/train"),
)

# Instantiate PyTorch Dataset
dataset = RetinalDataset(
    records=records,
    transform=transform_pipeline,
)
```

---

## Module Structure

```text
dataset/
├── __init__.py      # Public API exports
├── dataset.py        # RetinalDataset implementation
├── parser.py         # load_annotations parser implementation
├── types.py          # AnnotationRecord, DataSample dataclasses
├── exceptions.py     # DatasetError, InvalidAnnotationError
└── README.md        # Technical documentation
```

---

## Data Flow

```text
DatasetConfig / Annotation File
           │
           ▼
   load_annotations()
           │
           ▼
   [AnnotationRecord]
           │
           ▼
    RetinalDataset
           │
           ▼
      cv2.imread()
           │
           ▼
       RGB Image
           │
           ▼
       DataSample (image tensor, label)
```

---

## Design Principles

- **Single Responsibility Principle**: Only handles loading and serving single data samples.
- **Strong Typing**: Immutably typed via `AnnotationRecord` and `DataSample`.
- **Fail-Fast Parsing**: Validates file paths and label boundaries before dataset creation.
