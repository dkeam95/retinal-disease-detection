# Metrics Module (`src/metrics/`)

## Overview

The `metrics` module is dedicated to calculating classification and ordinal performance criteria. Since Diabetic Retinopathy grading is an ordinal problem, the primary clinical metric is **Quadratic Weighted Kappa (QWK)**, which heavily penalizes large margin misclassifications.

## Key Features

- **Clinical Alignment**: Primary evaluation uses QWK, penalizing errors according to distance (e.g., predicting 4 for a true class of 0 is penalized much more severely than predicting 1).
- **Comprehensive Evaluation**: Includes standard classification metrics: Accuracy, Precision, Recall, Macro-F1, and Confusion Matrix arrays.
- **Fail-Safe Shapes Checks**: Automatically validates predicted vs. target tensor dimensions before executing calculations.

## API & Interfaces

### Functions

- **`compute_quadratic_weighted_kappa(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 5) -> float`**
  Computes the Quadratic Weighted Kappa coefficient between target labels and predicted labels.
- **`compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float`**
  Computes accuracy score.
- **`compute_f1(y_true: np.ndarray, y_pred: np.ndarray, average: str = 'macro') -> float`**
  Computes F1-score.
- **`compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 5) -> np.ndarray`**
  Computes $num\_classes \times num\_classes$ confusion matrix.
- **`build_metrics(names: list[MetricName]) -> dict[MetricName, Callable]`**
  Factory building mapping of metric names to calculation functions.

### Exceptions

- `UnknownMetricError`: Raised if metric name requested is not supported.
- `MetricInitializationError`: Raised if shape verification fails.

## Dependencies

- `scikit-learn` (optional, or NumPy implementation): Calculation of kappa and f1 scores.
- `numpy`: Fast matrix operations.
