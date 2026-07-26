# Metrics Module

## Purpose

The `metrics` module provides a unified interface for evaluating model performance during validation and testing.

All metrics are implemented as independent functions and are managed through a registry/factory architecture. This allows metrics to be selected entirely through configuration without modifying the training pipeline.

---

## Architecture

```
metrics/
│
├── __init__.py
│
├── metric_names.py
├── exceptions.py
│
├── registry.py
├── factory.py
│
├── _validation.py
│
├── accuracy.py
├── precision.py
├── recall.py
├── f1.py
├── quadratic_weighted_kappa.py
├── confusion_matrix.py
│
└── README.md
```

---

## Supported Metrics

### Accuracy

Measures the proportion of correctly classified samples.

Returns:

- float

---

### Precision

Measures the proportion of correct positive predictions.

Supported averaging:

- macro
- weighted

Returns:

- float

---

### Recall

Measures the proportion of correctly identified positive samples.

Supported averaging:

- macro
- weighted

Returns:

- float

---

### F1 Score

Computes the harmonic mean of Precision and Recall.

Supported averaging:

- macro
- weighted

Returns:

- float

---

### Quadratic Weighted Kappa

Implementation of Cohen's Quadratic Weighted Kappa (QWK).

Recommended for ordinal classification problems such as Diabetic Retinopathy grading.

Returns:

- float

---

### Confusion Matrix

Computes the full confusion matrix.

Returns:

- torch.Tensor
- shape (C, C)

---

## Factory Pattern

Metrics are never called directly by the training pipeline.

Instead:

```python
metrics = build_metrics(metric_names)
```

returns a dictionary

```python
{
    MetricName.ACCURACY: compute_accuracy,
    MetricName.F1: compute_f1,
    ...
}
```

---

## Registry

All available metrics are registered inside

```
_METRIC_REGISTRY
```

using

```
MetricName
```

instead of raw strings.

Advantages:

- IDE autocompletion
- compile-time validation
- type safety
- centralized registration

---

## Validation

All metrics share the same validation utility.

```
validate_shapes()
```

Validates:

- logits shape
- target shape
- batch size consistency

---

## Exceptions

The module defines custom exceptions.

```
MetricError
```

Base exception.

```
UnknownMetricError
```

Raised when an unknown metric is requested.

```
MetricInitializationError
```

Raised when metric configuration or inputs are invalid.

---

## Testing

Covered by unit tests:

- metric names
- exceptions
- registry
- factory
- accuracy
- precision
- recall
- f1
- quadratic weighted kappa
- confusion matrix
- validation utilities

---

## Design Principles

- Single Responsibility Principle
- Open/Closed Principle
- Factory Pattern
- Registry Pattern
- Configuration Driven
- Type Safe API
- Shared Validation
- Modular Design
