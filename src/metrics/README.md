# Metrics Module

## Purpose

The `metrics` module provides a unified, type-safe evaluation interface for computing model performance during validation and testing.

Primary focus is placed on **Quadratic Weighted Kappa (QWK)**, the clinical gold standard metric for Diabetic Retinopathy severity grading.

---

## Supported Metrics

| Metric | Name Enum | Output Type | Description |
| :--- | :--- | :--- | :--- |
| **Accuracy** | `MetricName.ACCURACY` | `float` | Proportion of correctly classified samples. |
| **Precision** | `MetricName.PRECISION` | `float` | Positive predictive value (Macro / Weighted). |
| **Recall** | `MetricName.RECALL` | `float` | Sensitivity / True Positive Rate (Macro / Weighted). |
| **F1 Score** | `MetricName.F1` | `float` | Harmonic mean of Precision and Recall. |
| **Quadratic Weighted Kappa** | `MetricName.QWK` | `float` | Ordinal agreement score measuring distance between true and predicted grades. |
| **Confusion Matrix** | `MetricName.CONFUSION_MATRIX` | `torch.Tensor` | $5 \times 5$ confusion matrix array. |

---

## Why Quadratic Weighted Kappa (QWK)?

Diabetic Retinopathy grading is an **ordinal classification** task ($0 < 1 < 2 < 3 < 4$).

Unlike standard Accuracy or Cross-Entropy, QWK penalizes classification errors based on distance:
- Predicting Grade 1 for a Grade 0 patient incurs a minor distance penalty.
- Predicting Grade 4 for a Grade 0 patient incurs a quadratic distance penalty.

QWK is the primary metric used in clinical benchmarks (APTOS, EyePACS, Messidor, DDR).

---

## Module Structure

```text
metrics/
├── __init__.py                   # Public API exports
├── metric_names.py               # MetricName enumeration
├── factory.py                    # build_metrics factory function
├── registry.py                   # METRIC_REGISTRY mapping
├── accuracy.py                   # Accuracy metric
├── precision.py                  # Precision metric
├── recall.py                     # Recall metric
├── f1.py                         # F1 score metric
├── quadratic_weighted_kappa.py   # QWK calculation implementation
├── confusion_matrix.py           # Confusion matrix calculation
├── _validation.py                # Shape and tensor validation
├── exceptions.py                 # MetricError hierarchy
└── README.md                     # Technical documentation
```

---

## Public API Usage

```python
from metrics import build_metrics, MetricName

# Instantiate metrics registry
metric_fn_map = build_metrics([MetricName.ACCURACY, MetricName.QWK, MetricName.F1])

# Compute metrics
qwk_score = metric_fn_map[MetricName.QWK](logits, targets)
```

---

## Design Principles

- **Single Responsibility Principle**: Each metric is an isolated, functional unit.
- **Shared Validation**: Shape and boundary validation performed via `validate_shapes()`.
- **Enum-Driven Registration**: Avoids string typos through `MetricName` static enumeration.
