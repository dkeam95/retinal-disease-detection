# Dataset Module

## Purpose

The dataset module is responsible for loading dataset annotations,
reading images from disk, and providing samples through the PyTorch
Dataset interface.

---

## Responsibilities

This module is responsible only for:

- reading annotation files;
- validating annotation records;
- loading images from disk;
- creating dataset samples;
- exposing a PyTorch-compatible dataset.

This module does **not** perform:

- image preprocessing;
- data augmentation;
- tensor conversion;
- model creation;
- training;
- evaluation;
- inference;
- checkpoint management.

---

## Public API

```python
RetinalDataset(config: DatasetConfig)

load_annotations(
    annotation_file: Path,
    image_directory: Path,
) -> list[AnnotationRecord]
```

---

## Input

```text
DatasetConfig
```

---

## Output

```text
DataSample
```

---

## Dependencies

- torch
- opencv-python
- numpy
- common.config

---

## File Structure

```text
dataset/
├── __init__.py
├── dataset.py
├── parser.py
├── types.py
├── exceptions.py
└── README.md
```

---

## Workflow

```text
DatasetConfig
        │
        ▼
load_annotations()
        │
        ▼
AnnotationRecord
        │
        ▼
RetinalDataset
        │
        ▼
cv2.imread()
        │
        ▼
RGB image
        │
        ▼
DataSample
```

---

## Components

### parser.py

Reads annotation files and converts them into
`AnnotationRecord` objects.

---

### dataset.py

Implements the PyTorch Dataset interface and loads images
from disk.

---

### types.py

Contains immutable data structures shared across the dataset
module.

---

### exceptions.py

Defines dataset-specific exceptions.

---

## Design Principles

The module follows:

- Single Responsibility Principle (SRP)
- Configuration-driven dataset creation
- Strong typing
- Immutable data structures
- Separation of parsing and loading logic
- Minimal coupling

---

## Future Extensions

Future versions of this module may include:

- support for multiple annotation formats;
- dataset caching;
- lazy image loading;
- memory-mapped datasets;
- distributed dataset support;
- automatic integrity verification.
