"""Unit tests for ExperimentComparator and ExperimentRunner modules."""

from __future__ import annotations

import json

from experiments.comparator import ExperimentComparator


def test_experiment_comparator(tmp_path):
    # Setup dummy experiment output directories with metrics.json files
    exp1_dir = tmp_path / "exp_01"
    exp1_dir.mkdir()
    metrics1 = {
        "experiment_name": "exp_01",
        "architecture": "efficientnet_b0",
        "loss_function": "cross_entropy",
        "best_epoch": 5,
        "best_val_score": 0.82,
        "test_metrics": {"qwk": 0.81, "accuracy": 0.85, "macro_f1": 0.79},
    }
    with (exp1_dir / "metrics.json").open("w") as f:
        json.dump(metrics1, f)

    exp2_dir = tmp_path / "exp_02"
    exp2_dir.mkdir()
    metrics2 = {
        "experiment_name": "exp_02",
        "architecture": "convnext_tiny",
        "loss_function": "focal",
        "best_epoch": 8,
        "best_val_score": 0.87,
        "test_metrics": {"qwk": 0.86, "accuracy": 0.89, "macro_f1": 0.84},
    }
    with (exp2_dir / "metrics.json").open("w") as f:
        json.dump(metrics2, f)

    comparator = ExperimentComparator(experiments_root_dir=tmp_path)
    df = comparator.generate_leaderboard_dataframe()

    assert len(df) == 2
    assert "Experiment" in df.columns
    assert df.iloc[0]["Experiment"] == "exp_02"  # Sorted by QWK descending

    chart_file = tmp_path / "comparison.png"
    fig = comparator.plot_comparison_chart(save_path=chart_file)
    assert fig is not None
    assert chart_file.exists()

    html_file = tmp_path / "leaderboard.html"
    out_path = comparator.export_html_leaderboard(output_path=html_file)
    assert out_path.exists()
    assert "exp_02" in out_path.read_text(encoding="utf-8")
