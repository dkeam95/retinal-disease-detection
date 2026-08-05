# Evaluation Module (`src/evaluation`)

Evaluation engine and medical error analysis framework for Retinal Disease Detection models.

## 📌 Overview

The `src/evaluation` module provides tools to rigorously evaluate trained PyTorch checkpoints on validation/test datasets. It computes clinical metrics (QWK, F1, Sensitivity, Specificity, Confusion Matrix) and generates granular error analysis reports to inspect misclassifications.

---

## 🗂️ Directory Structure

```
src/evaluation/
├── __init__.py           # Module exports (Evaluator, ErrorAnalyzer)
├── eval.py               # Standalone CLI evaluation script
├── evaluator.py          # Core Evaluator engine for metrics calculation
└── error_analysis.py     # Advanced ErrorAnalyzer for diagnostic failure mode audits
```

---

## 🔑 Key Components

### 1. `Evaluator` (`evaluator.py`)

Computes comprehensive validation and test statistics across batches:

- Accuracy, Precision, Recall, Macro/Weighted F1 Score.
- **Quadratic Weighted Kappa (QWK)**: Primary medical metric for DR grading severity.
- Multi-class confusion matrices and normalized confusion heatmaps.

### 2. `ErrorAnalyzer` (`error_analysis.py`)

Diagnostic module for inspecting model failure modes:

- Extracts top misclassifications (false positives, false negatives).
- Analyzes off-by-one errors (e.g. Mild NPDR vs Moderate NPDR) vs severe off-by-N errors.
- Generates JSON and Markdown error reports with sample IDs for follow-up explainability (Grad-CAM).

### 3. CLI Evaluator (`eval.py`)

Entrypoint for evaluating checkpoints directly from command line arguments or config YAML.

---

## 💻 Usage Example

```python
from src.evaluation import ErrorAnalyzer, Evaluator

# Run full evaluation
evaluator = Evaluator(model=model, dataloader=test_loader, device=device)
results = evaluator.evaluate()

print(f"Test QWK: {results['qwk']:.4f}")
print(f"Test F1:  {results['f1_macro']:.4f}")

# Perform error analysis
analyzer = ErrorAnalyzer(results=results, save_dir="reports/error_analysis")
analyzer.generate_report()
```
