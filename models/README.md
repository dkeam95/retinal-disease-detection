# Models Directory (`models/`)

Directory structure for trained model checkpoints, exported artifacts, and pretrained weights.

## 📌 Overview

This directory stores neural network weights and serialized model artifacts.

> ⚠️ **Note**: Binary weight files (`*.ckpt`, `*.pth`, `*.pt`, `*.onnx`) are ignored by `.gitignore` to prevent uploading large binaries to GitHub.

---

## 🗂️ Directory Structure

```
models/
├── checkpoints/ # PyTorch Lightning / Trainer model checkpoints saved during training
├── exported/    # Serialized ONNX / TorchScript models for production inference
└── pretrained/  # Local cache for ImageNet / Med3D pretrained backbones
```
