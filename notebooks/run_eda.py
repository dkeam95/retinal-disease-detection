"""
Exploratory Data Analysis (EDA) script for Retinal Disease Dataset.

This module computes dataset distribution metrics, visualizes class balance,
analyzes image dimensions/channels, and displays sample images for each disease grade.
"""

import sys
from collections import Counter
from pathlib import Path

# -----------------------------------------------------------------------------
# 1. FIX PATHS & IMPORTS (Add src/ to sys.path to prevent import errors)
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from common.config.types import DatasetConfig
from dataset.dataset import RetinalDataset


def run_retinal_eda(config: DatasetConfig) -> None:
    """Run full EDA pipeline on retinal dataset."""

    print("=" * 60)
    print(" 1. DATASET LOADING AND CLASS BALANCE ANALYSIS ")
    print("=" * 60)

    # Initialize our PyTorch dataset
    dataset = RetinalDataset(config)
    total_samples = len(dataset)

    # Gather labels directly from the dataset's parsed annotations
    labels = [record.label for record in dataset._annotations]
    class_counts = Counter(labels)

    # Class names for Diabetic Retinopathy (Diabetic Retinopathy - DR)
    class_names = {
        0: "0 - No DR",
        1: "1 - Mild",
        2: "2 - Moderate",
        3: "3 - Severe",
        4: "4 - Proliferative",
    }

    print(f"Total images in dataset: {total_samples}\n")
    for label in sorted(class_counts.keys()):
        count = class_counts[label]
        percentage = (count / total_samples) * 100
        name = class_names.get(label, f"Class {label}")
        print(f"  {name:<20}: {count:>5} images ({percentage:>5.2f}%)")

    # Class distribution histogram
    plt.figure(figsize=(10, 5))
    sorted_labels = sorted(class_counts.keys())
    x_labels = [class_names.get(c, f"Class {c}") for c in sorted_labels]
    counts = [class_counts[c] for c in sorted_labels]

    bars = plt.bar(x_labels, counts, color="#3498db", edgecolor="black", alpha=0.85)
    plt.title("Distribution of Diabetic Retinopathy Severity Grades", fontsize=12, fontweight="bold")
    plt.ylabel("Number of Images")
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    # Add quantity labels above bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + (max(counts) * 0.01), f"{yval}", ha="center", va="bottom")

    plt.tight_layout()
    plt.show()

    print("\n" + "=" * 60)
    print(" 2. DIMENSION ANALYSIS AND MEAN/STD CALCULATION ")
    print("=" * 60)

    shapes = []
    means = []
    stds = []

    print("Scanning images via RetinalDataset...")
    for index in tqdm(range(total_samples), desc="Processing images"):
        sample = dataset[index]
        img = sample.image  # NumPy array (H, W, C) in RGB

        shapes.append(img.shape)

        # Normalize to [0.0, 1.0] for normalization statistics
        img_float = img / 255.0
        means.append(img_float.mean(axis=(0, 1)))
        stds.append(img_float.std(axis=(0, 1)))

    # Print resolution distribution
    unique_shapes = Counter(shapes)
    print("\nImage dimensions in dataset (Height, Width, Channels):")
    for shape, count in unique_shapes.items():
        print(f"  Format {shape}: {count} files")

    # Calculate dataset-wide statistics
    global_mean = np.mean(means, axis=0)
    global_std = np.mean(stds, axis=0)

    print("\n" + "-" * 50)
    print(" RECOMMENDED PYTORCH NORMALIZATION PARAMETERS:")
    print("-" * 50)
    print(f"  mean = [{global_mean[0]:.4f}, {global_mean[1]:.4f}, {global_mean[2]:.4f}]")
    print(f"  std  = [{global_std[0]:.4f}, {global_std[1]:.4f}, {global_std[2]:.4f}]")
    print("-" * 50)

    print("\n" + "=" * 60)
    print(" 3. VISUALIZATION OF SAMPLE IMAGES BY CLASS ")
    print("=" * 60)

    # Gather sample indices for each class
    class_indices = {c: [] for c in class_names.keys()}
    for idx, record in enumerate(dataset._annotations):
        if record.label in class_indices:
            class_indices[record.label].append(idx)

    # Display 2 examples per class
    num_classes = len(class_names)
    fig, axes = plt.subplots(num_classes, 2, figsize=(8, 3 * num_classes))

    for row, (cls_id, name) in enumerate(class_names.items()):
        indices = class_indices[cls_id][:2]  # First 2 examples
        for col in range(2):
            ax = axes[row, col] if num_classes > 1 else axes[col]
            if col < len(indices):
                idx = indices[col]
                sample = dataset[idx]
                ax.imshow(sample.image)
                ax.set_title(f"{name}\nIndex: {idx}", fontsize=9)
            ax.axis("off")

    plt.suptitle("Sample Fundus Images by Class", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    dataset_config = DatasetConfig(
        path=PROJECT_ROOT / "data" / "raw",
        annotation_file="train.txt",   # Annotation text file
        image_directory="train",       # Directory with images for training dataset
        num_classes=5,
    )

    run_retinal_eda(dataset_config)
