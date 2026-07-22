# Metrics Module

## Purpose

Provides evaluation metrics for diabetic retinopathy classification.

The module operates directly on model logits and ground-truth labels.

---

## Implemented Metrics

| Metric                   | Description                          | Medical relevance                          |
| ------------------------ | ------------------------------------ | ------------------------------------------ |
| Accuracy                 | Overall classification accuracy      | General quality                            |
| Precision                | Positive prediction quality          | False positive control                     |
| Recall                   | Sensitivity                          | False negative control                     |
| F1 Score                 | Balance between Precision and Recall | General classifier evaluation              |
| Quadratic Weighted Kappa | Ordinal agreement metric             | ⭐ Primary metric for diabetic retinopathy |

---

## Input

All metrics accept

```python
logits: Tensor (N, C)

targets: Tensor (N,)
```

---

## Output

Every metric returns

```python
float
```

---

## Validation

Every metric validates

- logits dimensions
- targets dimensions
- batch size consistency

---

## Unit Tests

Each metric contains

- perfect prediction test
- partial prediction test
- invalid input tests
- shape validation

Quadratic Weighted Kappa additionally contains

- deterministic evaluation

---

## Why Quadratic Weighted Kappa?

Unlike Accuracy, QWK considers the ordinal nature of diabetic retinopathy stages.

For example

```
0 → 1
```

is penalized much less than

```
0 → 4
```

making it the standard evaluation metric used in:

- APTOS
- EyePACS
- Messidor
- most DR research papers
