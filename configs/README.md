# Configuration Directory (`configs/`)

Hierarchical YAML configuration files and experiment overrides for Retinal Disease Detection.

## 📌 Overview

This directory contains the central configuration files governing dataset paths, image preprocessing options, model architecture selection, loss functions, optimizer settings, learning rate schedules, and experiment overrides.

---

## 🗂️ Directory Structure

```
configs/
├── config.yaml                     # Base configuration (default settings)
└── experiments/                    # Architecture-specific experiment overrides
    ├── exp_01_resnet50_v2.yaml          # ResNet-50 configuration override
    ├── exp_02_densenet121_v2.yaml       # DenseNet-121 configuration override
    ├── exp_03_efficientnet_b0_v2.yaml   # EfficientNet-B0 configuration override
    ├── exp_04_mobilenetv3_large_v2.yaml # MobileNetV3-Large configuration override
    ├── exp_05_convnext_tiny_v2.yaml     # ConvNeXt-Tiny configuration override
    ├── exp_06_vit_small_v2.yaml         # Vision Transformer (ViT-Small) override
    └── exp_07_swin_t_v2.yaml            # Swin Transformer (Swin-T) override
```

---

## ⚙️ Configuration Inheritance & Deep Merge

* **`config.yaml`**: Contains default parameters across all modules.
* **`configs/experiments/*.yaml`**: Override specific keys (e.g. `model.name`, `training.batch_size`, `training.learning_rate`) while inheriting unmentioned defaults from `config.yaml`.

---

## 💻 Running Experiments

Pass the experiment name to `--experiment`:

```bash
python -m src.trainer.train --experiment resnet50_v2
python -m src.trainer.train --experiment densenet121_v2
```
