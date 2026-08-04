"""CLI evaluation entry point for Retinal Disease Detection framework."""

from __future__ import annotations

import os
import sys

_script_dir = os.path.dirname(os.path.abspath(__file__))
_src_dir = os.path.dirname(_script_dir)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

import argparse
from pathlib import Path

from common.config.loader import ConfigLoader
from dataloader.factory import build_test_dataloader, build_validation_dataloader
from dataset.dataset import RetinalDataset
from evaluation.evaluator import ModelEvaluator

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retinal Disease Detection - Model Evaluation CLI"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to saved checkpoint file (.pt)",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["test", "valid"],
        default="test",
        help="Dataset split to evaluate",
    )
    parser.add_argument(
        "--report", type=Path, default=None, help="Output HTML report path"
    )
    parser.add_argument(
        "--json", type=Path, default=None, help="Output JSON results path"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("RETINAL DISEASE DETECTION - MODEL EVALUATION CLI")
    print("=" * 60)

    # 1. Load config & device
    config = ConfigLoader.load(args.config)
    device = (
        "cuda"
        if config.training.device == "cuda" and torch.cuda.is_available()
        else "cpu"
    )
    print(f"Config Path : {args.config}")
    print(f"Checkpoint  : {args.checkpoint}")
    print(f"Target Split: {args.split}")
    print(f"Device      : {device}")

    # 2. Build Dataset & DataLoader
    dataset_root = (
        config.dataset.path
        if config.dataset.path.is_absolute()
        else (PROJECT_ROOT / config.dataset.path).resolve()
    )
    annotation_file = f"{args.split}.txt"
    if not (dataset_root / annotation_file).exists():
        annotation_file = config.dataset.annotation_file

    ds = RetinalDataset(config=config.dataset, annotation_file=annotation_file)
    loader_builder = (
        build_test_dataloader if args.split == "test" else build_validation_dataloader
    )
    loader = loader_builder(dataset=ds, config=config.dataloader)

    # 3. Instantiate Evaluator & run
    evaluator = ModelEvaluator.from_checkpoint(
        checkpoint_path=args.checkpoint, config=config, device=device
    )
    report_path = (
        args.report
        or PROJECT_ROOT / "reports" / "metrics" / f"eval_{args.split}_report.html"
    )
    json_path = (
        args.json
        or PROJECT_ROOT / "reports" / "metrics" / f"eval_{args.split}_metrics.json"
    )

    print("\nRunning evaluation loop...")
    results = evaluator.evaluate(
        dataloader=loader,
        save_report_path=report_path,
        save_json_path=json_path,
        experiment_name=f"Evaluation_{args.split.upper()}",
    )

    print("\nEvaluation Completed!")
    print(f"QWK Score  : {results['qwk']:.4f}")
    print(f"Accuracy   : {results['accuracy']:.4f}")
    print(f"Macro F1   : {results['macro_f1']:.4f}")
    print(f"HTML Report: {report_path}")
    print(f"JSON Output: {json_path}")
    print("=" * 60)


if __name__ == "__main__":
    import torch

    main()
