# Experiments Module (`src/experiments`)

Experiment execution runner and cross-architecture benchmarking framework.

## 📌 Overview

The `src/experiments` module automates multi-model benchmarking for Retinal Disease Detection. It allows running systematic experiments across various neural network backbones (Convolutional Networks and Vision Transformers) and comparing their metrics (QWK, F1, Loss, inference latency).

---

## 🗂️ Directory Structure

```
src/experiments/
├── __init__.py           # Module exports (ExperimentRunner, ExperimentComparator)
├── comparator.py         # Benchmark comparative analytics and table generator
└── runner.py             # Automated experiment execution runner
```

---

## 🔑 Key Components

### 1. `ExperimentRunner` (`runner.py`)
Executes sequential training experiments:
* Loads base configuration (`config/config.yaml`) and deep-merges specific experiment YAMLs (e.g. `configs/experiments/exp_01_resnet50_v2.yaml`).
* Handles training execution, logging, checkpoint management, and error isolation.

### 2. `ExperimentComparator` (`comparator.py`)
Parses run logs and generates comparative reports:
* Extracts best epoch metrics (Val Loss, Val QWK, Macro F1, Accuracy).
* Generates comparative Markdown summary tables and CSV benchmarks.
* Plots model comparison bar charts and convergence curves.

---

## 💻 Usage Example

```python
from src.experiments import ExperimentComparator, ExperimentRunner

# Run specific experiment configs
runner = ExperimentRunner(config_dir="configs/experiments")
runner.run_all(experiments=["exp_01_resnet50_v2", "exp_02_densenet121_v2"])

# Compare results across all runs
comparator = ExperimentComparator(log_dir="logs")
df_summary = comparator.generate_summary_table()
comparator.plot_metric_comparison(metric="qwk", save_path="reports/benchmarks/qwk_comparison.png")
```
