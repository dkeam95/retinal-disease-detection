# Metrics Module

## Purpose

This module contains evaluation metrics used to assess diabetic retinopathy classification models.

All metrics operate on model logits `(N, C)` and ground-truth labels `(N,)`.

---

## Implemented Metrics

### Accuracy

File:

```text
accuracy.py
```

Computes overall classification accuracy.

Returns:

```text
float
```

---

### Precision

File:

```text
precision.py
```

Uses `sklearn.metrics.precision_score`.

Supported averaging strategies:

- macro
- weighted

Returns:

```text
float
```

---

### Recall

File:

```text
recall.py
```

Uses `sklearn.metrics.recall_score`.

Supported averaging strategies:

- macro
- weighted

Returns:

```text
float
```

---

### F1 Score

File:

```text
f1.py
```

Uses `sklearn.metrics.f1_score`.

Supported averaging strategies:

- macro
- weighted

Returns:

```text
float
```

---

## Validation

Every metric validates:

- logits shape
- targets shape
- batch size consistency

---

## Unit Tests

Every metric contains tests for:

- perfect predictions
- partially correct predictions
- invalid arguments
- shape validation

All implemented metrics are fully covered by unit tests.
