# Dataset Module (`src/dataset/`)

## Overview

The `dataset` module encapsulates data parsing, image disk loading, and wrapping DDR data splits into standard PyTorch `Dataset` representations. It bridges the gap between text-based annotations on disk and tensor batches.

## Key Features

- **Robust Annotation Parsing**: Parses tab or space-delimited text annotations (`train.txt`, `valid.txt`, etc.) containing `<image_path> <label>` pairs, verifying format and label ranges.
- **RGB Conversion**: Automatically reads images using OpenCV and converts them from BGR to RGB format, ensuring standard color representation.
- **Lesion Detection Support**: Includes parser and dataset representations for lesion bounding boxes and segmentations.

## API & Interfaces

### Classes

#### `RetinalDataset(Dataset)`
PyTorch dataset for image classification.
- **`__init__(config: DatasetConfig, transform: Optional[Callable] = None)`**: Sets up annotation paths and directories.
- **`__len__() -> int`**: Returns total samples.
- **`__getitem__(index: int) -> DataSample`**: Returns a `DataSample` containing the loaded and preprocessed image tensor/array and integer label.

#### `LesionDetectionDataset(Dataset)`
PyTorch dataset for localized lesion detection.
- **`__init__(config: DetectionDatasetConfig, transform: Optional[Callable] = None)`**: Initializes bounding box annotations.

#### `AnnotationRecord`
Immutable record representing raw parsed annotations:
- `image_path`: Path
- `label`: int

#### `DataSample`
The model input tuple structure:
- `image`: np.ndarray or torch.Tensor
- `label`: int

### Functions

- **`load_annotations(annotation_file: Path, image_directory: Path) -> list[AnnotationRecord]`**
  Helper function that scans the annotation file and validates that all image files exist before returning typed record models.

### Exceptions

- `DatasetNotFoundError`: Raised when the data directory does not exist.
- `AnnotationFileNotFoundError`: Raised when the split annotation file is missing.
- `InvalidAnnotationError`: Raised when annotation formats are incorrect or labels are out of range.
- `ImageLoadingError`: Raised when OpenCV fails to read an image.

## Dependencies

- `opencv-python`: Disk image loading (`cv2.imread`).
- `torch`: PyTorch Dataset class.
