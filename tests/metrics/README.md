# Test Suite: Metrics Module

## Purpose

This test suite validates numerical accuracy, edge cases, and shape validation rules for all evaluation metrics in `src/metrics/`.

---

## Test Coverage Summary

- **`test_accuracy.py`**: Perfect predictions, 0% accuracy predictions, batch size mismatches.
- **`test_precision_recall_f1.py`**: Multi-class macro and weighted averaging, zero division handling.
- **`test_qwk.py`**: Perfect ordinal agreement (QWK = 1.0), random chance agreement (QWK $\approx$ 0.0), worst quadratic disagreement (QWK $< 0$).
- **`test_confusion_matrix.py`**: $C \times C$ shape validation ($5 \times 5$), diagonal counts for ground truth matches.
- **`test_validation.py`**: Input shape mismatch detection, non-2D logits detection, non-1D targets detection.

---

## Running Tests

Execute pytest specifically for metric tests:

```bash
pytest tests/metrics
```
