"""CLI Prediction Entry Point for Retinal Disease Detection."""

from __future__ import annotations

import os
import sys

_script_dir = os.path.dirname(os.path.abspath(__file__))
_src_dir = os.path.dirname(_script_dir)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

import argparse
import json
from pathlib import Path

from common.config.loader import ConfigLoader
from inference.predictor import RetinalPredictor

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description="Retinal Disease Detection - Single Image Prediction CLI")
    parser.add_argument("--image", type=Path, required=True, help="Path to input fundus image (.jpg, .png)")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to trained PyTorch model checkpoint (.pt)")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "config.yaml", help="Path to configuration file")
    args = parser.parse_args()

    print("=" * 60)
    print("RETINAL DISEASE DETECTION - INFERENCE CLI")
    print("=" * 60)

    # Load config and initialize predictor
    config = ConfigLoader.load(args.config)
    device = "cuda" if config.training.device == "cuda" and torch.cuda.is_available() else "cpu"

    print(f"Config     : {args.config}")
    print(f"Checkpoint : {args.checkpoint}")
    print(f"Input Image: {args.image}")
    print(f"Compute Dev: {device}")

    predictor = RetinalPredictor.from_checkpoint(checkpoint_path=args.checkpoint, config=config, device=device)

    print("\nRunning inference...")
    result = predictor.predict(args.image)

    print("\nPrediction Result:")
    print(f"  Predicted Grade : {result.grade_id} ({result.grade_name})")
    print(f"  Confidence      : {result.confidence * 100:.2f}%")
    print("  Probability Distribution:")
    for i, prob in enumerate(result.probabilities):
        name = predictor.class_names[i] if i < len(predictor.class_names) else f"Grade {i}"
        print(f"    - Grade {i} ({name:16s}): {prob * 100:6.2f}%")

    print("=" * 60)


if __name__ == "__main__":
    import torch
    main()
