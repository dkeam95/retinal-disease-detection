"""Experiment Runner module for multi-model & multi-loss systematic studies."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from common.config.loader import ConfigLoader
from trainer.train import (
    build_dataloaders,
    build_datasets,
    build_trainer,
    build_training_components,
    resolve_device,
    set_seed,
)


class ExperimentRunner:
    """Orchestrates systematic experiments across multiple YAML configs."""

    def __init__(self, experiment_configs_dir: str | Path, output_dir: str | Path = "experiments") -> None:
        """Initialize ExperimentRunner.

        Args:
            experiment_configs_dir: Directory containing YAML experiment config files.
            output_dir: Parent output directory for saving experiment results.
        """
        self.configs_dir = Path(experiment_configs_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_all(self, max_epochs_override: int | None = None) -> list[dict[str, Any]]:
        """Discover and execute all YAML configs in experiment_configs_dir.

        Args:
            max_epochs_override: Optional epoch limit for fast experimental runs.

        Returns:
            List of result summary dicts for all completed experiments.
        """
        config_files = sorted(list(self.configs_dir.glob("*.yaml")) + list(self.configs_dir.glob("*.yml")))
        if not config_files:
            raise FileNotFoundError(f"No YAML config files found in directory: {self.configs_dir}")

        results_list = []

        for config_path in config_files:
            print("\n" + "=" * 60)
            print(f"STARTING EXPERIMENT: {config_path.name}")
            print("=" * 60)

            res = self.run_single(config_path, max_epochs_override=max_epochs_override)
            results_list.append(res)

        return results_list

    def run_single(self, config_path: str | Path, max_epochs_override: int | None = None) -> dict[str, Any]:
        """Run training and testing for a single experiment config file.

        Args:
            config_path: Path to YAML config file.
            max_epochs_override: Optional epoch override.

        Returns:
            Dictionary containing metrics and config summary for the run.
        """
        path = Path(config_path)
        config = ConfigLoader.load(path)

        if max_epochs_override is not None:
            # Create modified copy with epoch override
            from dataclasses import replace
            config = replace(config, training=replace(config.training, epochs=max_epochs_override))

        exp_name = config.experiment.name
        exp_dir = self.output_dir / exp_name
        exp_dir.mkdir(parents=True, exist_ok=True)

        set_seed(config.experiment.seed)
        device = resolve_device(config.training.device)

        train_ds, val_ds, test_ds = build_datasets(config)
        train_loader, val_loader, test_loader = build_dataloaders(config, train_ds, val_ds, test_ds)

        model, criterion, optimizer, scheduler = build_training_components(config)

        # Update checkpoint directory to experiment output dir
        from dataclasses import replace
        config = replace(config, checkpoint=replace(config.checkpoint, directory=exp_dir / "checkpoints"))

        trainer = build_trainer(
            config=config,
            train_loader=train_loader,
            validation_loader=val_loader,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
        )

        summary = trainer.fit(epochs=config.training.epochs, monitor=config.checkpoint.monitor)
        test_epoch_output = trainer.test(test_loader)
        test_metrics_dict = {"test_loss": float(test_epoch_output.loss), **test_epoch_output.metrics}

        result = {
            "experiment_name": exp_name,
            "architecture": config.model.architecture,
            "loss_function": config.loss.name,
            "best_epoch": summary.best_epoch,
            "best_val_score": float(summary.best_metric),
            "test_metrics": test_metrics_dict,
        }

        # Save metrics json artifact in experiment directory
        with (exp_dir / "metrics.json").open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

        return result


def main() -> None:
    import argparse
    import os
    import sys

    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _src_dir = os.path.dirname(_script_dir)
    if _src_dir not in sys.path:
        sys.path.insert(0, _src_dir)

    from experiments.comparator import ExperimentComparator

    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Retinal Disease Detection - Experiment Runner CLI")
    parser.add_argument("--configs_dir", type=Path, default=project_root / "configs" / "experiments", help="Directory containing experiment YAML configs")
    parser.add_argument("--output_dir", type=Path, default=project_root / "experiments", help="Output directory for experiment runs")
    parser.add_argument("--epochs", type=int, default=None, help="Optional epoch limit override for all experiments")
    args = parser.parse_args()

    print("=" * 60)
    print("RETINAL DISEASE DETECTION - EXPERIMENT RUNNER")
    print("=" * 60)
    print(f"Configs Dir: {args.configs_dir}")
    print(f"Output Dir : {args.output_dir}")

    runner = ExperimentRunner(experiment_configs_dir=args.configs_dir, output_dir=args.output_dir)
    results = runner.run_all(max_epochs_override=args.epochs)

    print("\nAll Experiments Completed!")
    print(f"Total Runs: {len(results)}")

    print("\nExporting Experiments Leaderboard Dashboard...")
    comparator = ExperimentComparator(experiments_root_dir=args.output_dir)
    leaderboard_file = project_root / "reports" / "metrics" / "experiments_leaderboard.html"
    comparator.export_html_leaderboard(output_path=leaderboard_file)

    print(f"Leaderboard exported to: {leaderboard_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
