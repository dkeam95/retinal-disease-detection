"""CLI script for batch generating Grad-CAM heatmaps for trained models and experiments."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import torch

_script_dir = os.path.dirname(os.path.abspath(__file__))
_src_dir = os.path.dirname(_script_dir)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from common.config.loader import ConfigLoader
from explainability.gradcam import GradCAM
from preprocessing.pipeline import build_validation_pipeline
from training.model_factory import ModelFactory

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def generate_experiment_heatmaps(
    checkpoint_path: str | Path,
    config_path: str | Path,
    images_dir: str | Path,
    output_dir: str | Path = PROJECT_ROOT / "reports" / "figures" / "gradcam",
    experiment_name: str | None = None,
    max_images: int = 10,
    device: str = "cpu",
) -> list[Path]:
    """Generate Grad-CAM heatmaps for images in a directory using a model checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint (.pt).
        config_path: Path to YAML experiment config.
        images_dir: Directory containing input fundus images.
        output_dir: Parent output directory for saving Grad-CAM heatmaps.
        experiment_name: Optional experiment name override.
        max_images: Maximum number of images to process.
        device: Computing device ('cpu', 'cuda').

    Returns:
        List of Path objects to saved heatmap images.
    """
    ckpt_path = Path(checkpoint_path)
    cfg_path = Path(config_path)
    img_dir = Path(images_dir)
    out_dir = Path(output_dir)

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    if not img_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {img_dir}")

    config = ConfigLoader.load(cfg_path)
    dev = torch.device(
        device if torch.cuda.is_available() and device == "cuda" else "cpu"
    )

    exp_name = experiment_name or config.experiment.name or config.model.architecture

    # Build model & load state dict
    model = ModelFactory.build(
        architecture=config.model.architecture,
        pretrained=False,
        num_classes=config.model.num_classes,
        dropout_rate=config.model.dropout_rate,
    )
    checkpoint_data = torch.load(ckpt_path, map_location=dev, weights_only=False)
    state_dict = (
        checkpoint_data["model_state"]
        if isinstance(checkpoint_data, dict) and "model_state" in checkpoint_data
        else checkpoint_data
    )
    model.load_state_dict(state_dict)
    model.to(dev)

    transform = build_validation_pipeline(config.preprocessing)
    gradcam = GradCAM(model=model)

    image_files = sorted(
        [p for p in img_dir.glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
    )[:max_images]
    saved_paths = []

    print(f"Generating Grad-CAM heatmaps for experiment: {exp_name}")
    print(f"Model Architecture: {config.model.architecture}")
    print(f"Output Subfolder   : {out_dir / exp_name}")
    print(f"Processing {len(image_files)} images...")

    for img_path in image_files:
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            continue
        original_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        augmented = transform(image=original_rgb)
        tensor = augmented["image"].unsqueeze(0).to(dev)

        # Get model prediction
        model.eval()
        with torch.no_grad():
            outputs = model(tensor)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs
            probs = torch.softmax(logits, dim=1).squeeze(0)
            pred_class = int(torch.argmax(probs).item())
            confidence = float(probs[pred_class].item())

        saved_path = gradcam.save_experiment_visualization(
            input_tensor=tensor,
            original_rgb=original_rgb,
            image_id=img_path.name,
            experiment_or_model=exp_name,
            output_dir=out_dir,
            predicted_label=pred_class,
            confidence=confidence,
            target_class=pred_class,
        )
        saved_paths.append(saved_path)
        print(f"  [OK] Saved: {saved_path.name}")

    return saved_paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retinal Disease Detection - Grad-CAM Batch Generator"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to trained PyTorch model checkpoint (.pt)",
    )
    parser.add_argument(
        "--config", type=Path, required=True, help="Path to experiment YAML config"
    )
    parser.add_argument(
        "--images_dir",
        type=Path,
        required=True,
        help="Directory containing fundus images",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=PROJECT_ROOT / "reports" / "figures" / "gradcam",
        help="Output directory for Grad-CAM figures",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="Optional experiment name override",
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=10,
        help="Maximum number of images to visualize",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Compute device ('cuda' or 'cpu')"
    )
    args = parser.parse_args()

    generate_experiment_heatmaps(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        images_dir=args.images_dir,
        output_dir=args.output_dir,
        experiment_name=args.experiment_name,
        max_images=args.max_images,
        device=args.device,
    )


if __name__ == "__main__":
    main()
