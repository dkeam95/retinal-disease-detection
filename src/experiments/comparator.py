"""Experiment Comparator for multi-model & multi-loss ablation studies."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


class ExperimentComparator:
    """Aggregates metrics across completed experiment runs and generates comparative leaderboards and charts."""

    def __init__(self, experiments_root_dir: str | Path = "experiments") -> None:
        """Initialize Comparator.

        Args:
            experiments_root_dir: Root directory containing individual experiment folders with metrics.json.
        """
        self.root_dir = Path(experiments_root_dir)

    def load_all_experiments(self) -> list[dict[str, Any]]:
        """Find and parse all metrics.json files under experiments root directory.

        Returns:
            List of experiment summary dictionaries.
        """
        json_files = list(self.root_dir.glob("**/metrics.json"))
        if not json_files:
            return []

        experiments = []
        for path in json_files:
            try:
                with path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                    experiments.append(data)
            except Exception as e:
                print(f"Warning: Failed to parse experiment metrics at {path}: {e}")

        # Sort by QWK descending if available
        experiments.sort(
            key=lambda x: x.get("test_metrics", {}).get("qwk", x.get("best_val_score", 0.0)),
            reverse=True,
        )
        return experiments

    def generate_leaderboard_dataframe(self) -> pd.DataFrame:
        """Construct a pandas DataFrame summary leaderboard table of all experiments.

        Returns:
            DataFrame with columns: Experiment, Architecture, Loss Function, Best Epoch, Val Score, Test QWK, Test Accuracy, Test Macro F1.
        """
        experiments = self.load_all_experiments()
        rows = []
        for exp in experiments:
            tm = exp.get("test_metrics", {})
            rows.append({
                "Experiment": exp.get("experiment_name", "N/A"),
                "Architecture": exp.get("architecture", "N/A"),
                "Loss Function": exp.get("loss_function", "N/A"),
                "Best Epoch": exp.get("best_epoch", 0),
                "Best Val Score": exp.get("best_val_score", 0.0),
                "Test QWK": tm.get("qwk", tm.get("quadratic_weighted_kappa", 0.0)),
                "Test Accuracy": tm.get("accuracy", 0.0),
                "Test Macro F1": tm.get("macro_f1", tm.get("f1", 0.0)),
            })
        return pd.DataFrame(rows)

    def plot_comparison_chart(
        self,
        metric_name: str = "Test QWK",
        save_path: str | Path | None = None,
        title: str = "Model & Loss Function Performance Comparison",
    ) -> plt.Figure:
        """Generate a comparative horizontal bar chart across all completed experiments.

        Args:
            metric_name: Column metric name to compare.
            save_path: Optional output path for PNG figure.
            title: Chart title.

        Returns:
            Matplotlib Figure object.
        """
        df = self.generate_leaderboard_dataframe()
        if df.empty:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.text(0.5, 0.5, "No experiment results found", ha="center", va="center")
            return fig

        df_sorted = df.sort_values(by=metric_name, ascending=True)

        fig, ax = plt.subplots(figsize=(10, max(4, len(df_sorted) * 0.8)))
        labels = [f"{row['Experiment']}\n({row['Architecture']} | {row['Loss Function']})" for _, row in df_sorted.iterrows()]
        values = df_sorted[metric_name].values

        bars = ax.barh(labels, values, color="#1f77b4", edgecolor="#114b75", height=0.55)

        ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel(metric_name, fontsize=11)
        ax.set_xlim(0.0, 1.05)
        ax.grid(True, axis="x", linestyle="--", alpha=0.6)

        # Add bar numeric annotations
        for bar in bars:
            width = bar.get_width()
            ax.annotate(
                f"{width:.4f}",
                xy=(width, bar.get_y() + bar.get_height() / 2),
                xytext=(5, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=9,
                fontweight="bold",
            )

        plt.tight_layout()

        if save_path is not None:
            path = Path(save_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=300, bbox_inches="tight")

        return fig

    def export_html_leaderboard(self, output_path: str | Path = "reports/metrics/experiments_leaderboard.html") -> Path:
        """Generate an HTML dashboard summarizing all experiment runs.

        Args:
            output_path: Path to output HTML file.

        Returns:
            Path to saved HTML file.
        """
        df = self.generate_leaderboard_dataframe()
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)

        chart_fig = self.plot_comparison_chart(metric_name="Test QWK")
        from visualization.report_generator import figure_to_base64
        chart_b64 = figure_to_base64(chart_fig)

        table_html = df.to_html(index=False, classes="styled-table", float_format=lambda x: f"{x:.4f}")

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Retinal Disease Detection - Experiments Leaderboard</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #f4f7f9; padding: 20px; color: #333; }}
        .container {{ max-width: 1100px; margin: 0 auto; background: #fff; padding: 30px; border-radius: 10px; box-shadow: 0 4px 15px rgba(0,0,0,0.08); }}
        h1 {{ color: #1f3a60; border-bottom: 3px solid #2b5c8f; padding-bottom: 10px; }}
        .img-box {{ text-align: center; margin: 25px 0; background: #fafafa; padding: 15px; border: 1px solid #eee; border-radius: 6px; }}
        .img-box img {{ max-width: 100%; height: auto; }}
        .styled-table {{ width: 100%; border-collapse: collapse; margin-top: 15px; }}
        .styled-table th, .styled-table td {{ padding: 10px 14px; border-bottom: 1px solid #e0e0e0; text-align: left; }}
        .styled-table th {{ background: #1f3a60; color: #fff; }}
        .styled-table tr:nth-child(even) {{ background: #f9fbfd; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Retinal Disease Detection - Experiments Leaderboard</h1>
        <p>Comparative summary of multi-backbone and loss function ablation studies.</p>
        
        <div class="img-box">
            <img src="{chart_b64}" alt="Experiments Leaderboard Chart">
        </div>

        <h2>🏆 Leaderboard Table</h2>
        {table_html}
    </div>
</body>
</html>
"""
        with out_file.open("w", encoding="utf-8") as f:
            f.write(html_content)

        return out_file
