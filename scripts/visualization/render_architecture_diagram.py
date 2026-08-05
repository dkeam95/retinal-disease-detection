"""
Render Codebase Architecture & Module Dependency Diagram to High-Resolution PNG.

Generates reports/figures/codebase_architecture_diagram.png.
"""

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


def render_architecture_diagram(output_path: Path) -> None:
    """Render a clean, high-resolution architecture diagram of the retinal disease detection system."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 10), dpi=300)
    ax.set_facecolor("#f8fafc")
    fig.patch.set_facecolor("#f8fafc")

    # Define module boxes: (x, y, width, height, title, subtitle, color, border_color)
    modules = [
        (0.5, 8.5, 3.0, 1.0, "src/common", "Config, Classes, Exceptions", "#e0f2fe", "#0284c7"),
        (4.5, 8.5, 3.0, 1.0, "src/dataset", "Parsers, RetinalDataset", "#fef3c7", "#d97706"),
        (8.5, 8.5, 3.0, 1.0, "src/dataloader", "Batches, Albumentations", "#fef3c7", "#d97706"),

        (0.5, 6.0, 3.0, 1.2, "src/model", "ConvNeXt-Tiny, Swin-T,\nDenseNet121, ModelRegistry", "#dcfce7", "#16a34a"),
        (4.5, 6.0, 3.0, 1.2, "src/losses", "CB-Focal Loss,\nWeighted Cross-Entropy", "#fee2e2", "#dc2626"),
        (8.5, 6.0, 3.0, 1.2, "src/metrics", "QWK, F1-Score, Accuracy,\nConfusion Matrix", "#e0e7ff", "#4f46e5"),

        (4.5, 3.5, 4.0, 1.3, "src/trainer", "Epoch Loops, EarlyStopping,\nCheckpoint Manager", "#fae8ff", "#c084fc"),

        (0.5, 1.0, 3.2, 1.2, "src/detection", "Faster R-CNN (1536px),\nMicro-Anchors, mAP", "#fce7f3", "#db2777"),
        (4.2, 1.0, 3.2, 1.2, "src/preprocessing", "FundusFOVExtractor (98%),\nOpticDiscDetector", "#ccfbf1", "#0d9488"),
        (8.0, 1.0, 3.2, 1.2, "src/inference", "Ensemble (QWK=0.7685),\nMaskedLesionPredictor", "#ffedd5", "#ea580c"),
        (11.5, 1.0, 2.2, 1.2, "src/explainability", "5 SOTA XAI\nAlgorithms", "#fef08a", "#ca8a04"),
    ]

    # Draw module cards
    for x, y, w, h, title, subtitle, bg_color, border_color in modules:
        rect = mpatches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.15,rounding_size=0.2",
            facecolor=bg_color,
            edgecolor=border_color,
            linewidth=2,
        )
        ax.add_patch(rect)

        ax.text(
            x + w / 2.0, y + h * 0.65, title,
            ha="center", va="center",
            fontsize=12, fontweight="bold", color="#0f172a"
        )
        ax.text(
            x + w / 2.0, y + h * 0.30, subtitle,
            ha="center", va="center",
            fontsize=9, color="#475569"
        )

    # Draw arrows indicating data flow: (x1, y1, x2, y2)
    arrows = [
        (3.5, 9.0, 4.5, 9.0),   # common -> dataset
        (7.5, 9.0, 8.5, 9.0),   # dataset -> dataloader
        (10.0, 8.5, 6.5, 4.8),  # dataloader -> trainer
        (2.0, 6.0, 4.5, 4.2),   # model -> trainer
        (6.0, 6.0, 6.0, 4.8),   # losses -> trainer
        (10.0, 6.0, 7.5, 4.8),  # metrics -> trainer
        (2.0, 6.0, 9.0, 2.2),   # model -> inference
        (2.1, 1.0, 8.0, 1.6),   # detection -> inference
        (5.8, 1.0, 8.0, 1.6),   # preprocessing -> inference
        (11.2, 1.6, 11.5, 1.6), # inference -> explainability
    ]

    for x1, y1, x2, y2 in arrows:
        ax.annotate(
            "",
            xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(
                arrowstyle="-|>",
                color="#64748b",
                lw=2.0,
                mutation_scale=15,
            ),
        )

    ax.set_xlim(-0.2, 14.2)
    ax.set_ylim(-0.2, 10.5)
    ax.axis("off")

    plt.title(
        "Retinal Disease Detection — System Architecture & Module Dependency Diagram",
        fontsize=15, fontweight="bold", pad=20, color="#0f172a"
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Successfully rendered architecture diagram to: {output_path}")


if __name__ == "__main__":
    out_img = Path("reports/figures/codebase_architecture_diagram.png")
    render_architecture_diagram(out_img)
