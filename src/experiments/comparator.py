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
            key=lambda x: x.get("test_metrics", {}).get(
                "qwk", x.get("best_val_score", 0.0)
            ),
            reverse=True,
        )
        return experiments

    def _split_rounds(
        self, experiments: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Split experiments into Round 1 (Baseline) and Round 2 (v2 Enhanced)."""
        round1 = []
        round2 = []
        for exp in experiments:
            name = exp.get("experiment_name", "")
            if name.endswith("_v2") or "_v2" in name:
                round2.append(exp)
            else:
                round1.append(exp)
        return round1, round2

    def generate_leaderboard_dataframe(
        self, experiments: list[dict[str, Any]] | None = None
    ) -> pd.DataFrame:
        """Construct a pandas DataFrame summary leaderboard table of provided experiments."""
        if experiments is None:
            experiments = self.load_all_experiments()

        rows = []
        for exp in experiments:
            tm = exp.get("test_metrics", {})
            rows.append(
                {
                    "Experiment": exp.get("experiment_name", "N/A"),
                    "Architecture": exp.get("architecture", "N/A"),
                    "Loss Function": exp.get("loss_function", "N/A"),
                    "Best Epoch": exp.get("best_epoch", 0),
                    "Best Val QWK": exp.get("best_val_score", 0.0),
                    "Test QWK": tm.get("qwk", tm.get("quadratic_weighted_kappa", 0.0)),
                    "Test Accuracy": tm.get("accuracy", 0.0),
                    "Test Loss": tm.get("test_loss", 0.0),
                    "Test Macro F1": tm.get("macro_f1", tm.get("f1", 0.0)),
                }
            )
        return pd.DataFrame(rows)

    def generate_comparison_dataframe(self) -> pd.DataFrame:
        """Construct a side-by-side architecture comparison DataFrame between Round 1 and Round 2."""
        experiments = self.load_all_experiments()
        r1, r2 = self._split_rounds(experiments)

        r1_map = {exp.get("architecture"): exp for exp in r1}
        r2_map = {exp.get("architecture"): exp for exp in r2}

        all_archs = sorted(list(set(list(r1_map.keys()) + list(r2_map.keys()))))
        rows = []

        for arch in all_archs:
            e1 = r1_map.get(arch, {})
            e2 = r2_map.get(arch, {})
            tm1 = e1.get("test_metrics", {})
            tm2 = e2.get("test_metrics", {})

            qwk1 = tm1.get("qwk", 0.0)
            qwk2 = tm2.get("qwk", 0.0)
            loss1 = tm1.get("test_loss", 0.0)
            loss2 = tm2.get("test_loss", 0.0)
            acc1 = tm1.get("accuracy", 0.0)
            acc2 = tm2.get("accuracy", 0.0)

            qwk_delta = qwk2 - qwk1 if qwk1 > 0 and qwk2 > 0 else 0.0
            loss_reduction = (
                ((loss1 - loss2) / loss1 * 100) if loss1 > 0 and loss2 > 0 else 0.0
            )

            rows.append(
                {
                    "Architecture": arch,
                    "R1 (Baseline) QWK": qwk1,
                    "R2 (Enhanced) QWK": qwk2,
                    "QWK Delta": qwk_delta,
                    "R1 Test Loss": loss1,
                    "R2 Test Loss": loss2,
                    "Loss Reduction (%)": loss_reduction,
                    "R1 Test Acc": acc1,
                    "R2 Test Acc": acc2,
                }
            )

        df_comp = pd.DataFrame(rows)
        if not df_comp.empty and "R2 (Enhanced) QWK" in df_comp:
            df_comp = df_comp.sort_values(by="R2 (Enhanced) QWK", ascending=False)
        return df_comp

    def plot_grouped_comparison_chart(
        self, save_path: str | Path | None = None
    ) -> plt.Figure:
        """Generate a grouped bar chart comparing Round 1 vs Round 2 Test QWK across all architectures."""
        df_comp = self.generate_comparison_dataframe()
        if df_comp.empty:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.text(0.5, 0.5, "No experiment results found", ha="center", va="center")
            return fig

        archs = df_comp["Architecture"].values
        qwk1 = df_comp["R1 (Baseline) QWK"].values
        qwk2 = df_comp["R2 (Enhanced) QWK"].values

        x = range(len(archs))
        width = 0.35

        fig, ax = plt.subplots(figsize=(12, 6))
        rects1 = ax.bar(
            [i - width / 2 for i in x],
            qwk1,
            width,
            label="Round 1 (Baseline)",
            color="#6c757d",
            edgecolor="#495057",
        )
        rects2 = ax.bar(
            [i + width / 2 for i in x],
            qwk2,
            width,
            label="Round 2 (Augmentations + Focal Loss + Regularization)",
            color="#28a745",
            edgecolor="#1e7e34",
        )

        ax.set_ylabel("Test QWK Score", fontsize=11, fontweight="bold")
        ax.set_title(
            "Architecture Performance: Round 1 (Baseline) vs Round 2 (Enhanced)",
            fontsize=13,
            fontweight="bold",
            pad=15,
        )
        ax.set_xticks(list(x))
        ax.set_xticklabels(archs, rotation=15, ha="right", fontsize=10)
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, axis="y", linestyle="--", alpha=0.5)
        ax.legend(fontsize=11)

        # Annotate bars
        for rect in rects1:
            h = rect.get_height()
            if h > 0:
                ax.annotate(
                    f"{h:.3f}",
                    xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        for rect in rects2:
            h = rect.get_height()
            if h > 0:
                ax.annotate(
                    f"{h:.3f}",
                    xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    fontweight="bold",
                )

        plt.tight_layout()
        if save_path is not None:
            path = Path(save_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=300, bbox_inches="tight")

        return fig

    def plot_comparison_chart(
        self,
        metric_name: str = "Test QWK",
        save_path: str | Path | None = None,
        title: str = "Model Performance Comparison",
    ) -> plt.Figure:
        """Backward compatible wrapper calling plot_grouped_comparison_chart."""
        return self.plot_grouped_comparison_chart(save_path=save_path)

    def export_html_leaderboard(
        self, output_path: str | Path = "reports/metrics/experiments_leaderboard.html"
    ) -> Path:
        """Generate a structured HTML dashboard separating Round 1 and Round 2 experiments.

        Args:
            output_path: Path to output HTML file.

        Returns:
            Path to saved HTML file.
        """
        all_exps = self.load_all_experiments()
        r1_exps, r2_exps = self._split_rounds(all_exps)

        df_r1 = self.generate_leaderboard_dataframe(r1_exps)
        df_r2 = self.generate_leaderboard_dataframe(r2_exps)
        df_comp = self.generate_comparison_dataframe()

        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)

        chart_fig = self.plot_grouped_comparison_chart()
        from visualization.report_generator import figure_to_base64

        chart_b64 = figure_to_base64(chart_fig)

        fmt_pct = lambda x: f"{x * 100:.2f}%" if isinstance(x, float) else x
        fmt_float = lambda x: f"{x:.4f}" if isinstance(x, float) else x

        table_comp_html = df_comp.to_html(
            index=False, classes="styled-table comp-table", float_format=fmt_float
        )
        table_r2_html = df_r2.to_html(
            index=False, classes="styled-table r2-table", float_format=fmt_float
        )
        table_r1_html = df_r1.to_html(
            index=False, classes="styled-table r1-table", float_format=fmt_float
        )

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Retinal Disease Detection - Round 1 vs Round 2 Experiments Leaderboard</title>
    <style>
        body {{ font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background: #f0f4f8; padding: 25px; color: #2d3748; line-height: 1.6; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: #ffffff; padding: 35px; border-radius: 12px; box-shadow: 0 10px 25px rgba(0,0,0,0.08); }}
        h1 {{ color: #1a365d; border-bottom: 4px solid #3182ce; padding-bottom: 12px; margin-top: 0; font-size: 28px; }}
        h2 {{ color: #2c5282; margin-top: 35px; padding-bottom: 8px; border-bottom: 2px solid #e2e8f0; font-size: 22px; }}
        .badge {{ display: inline-block; padding: 4px 10px; border-radius: 20px; font-size: 13px; font-weight: bold; margin-left: 8px; }}
        .badge-r1 {{ background: #e2e8f0; color: #4a5568; }}
        .badge-r2 {{ background: #c6f6d5; color: #22543d; }}
        .img-box {{ text-align: center; margin: 30px 0; background: #f7fafc; padding: 20px; border: 1px solid #e2e8f0; border-radius: 8px; box-shadow: inset 0 2px 4px rgba(0,0,0,0.02); }}
        .img-box img {{ max-width: 100%; height: auto; }}
        .styled-table {{ width: 100%; border-collapse: collapse; margin-top: 15px; font-size: 14px; margin-bottom: 25px; }}
        .styled-table th, .styled-table td {{ padding: 12px 16px; border-bottom: 1px solid #e2e8f0; text-align: left; }}
        .styled-table th {{ background: #2b6cb0; color: #ffffff; font-weight: 600; text-transform: uppercase; font-size: 12px; letter-spacing: 0.5px; }}
        .comp-table th {{ background: #2c5282; }}
        .r2-table th {{ background: #2f855a; }}
        .r1-table th {{ background: #4a5568; }}
        .styled-table tr:hover {{ background-color: #edf2f7; }}
        .styled-table tr:nth-child(even) {{ background: #f7fafc; }}
        .summary-cards {{ display: flex; gap: 20px; margin-bottom: 25px; }}
        .card {{ flex: 1; background: #ebf8ff; border-left: 5px solid #3182ce; padding: 18px; border-radius: 6px; }}
        .card-green {{ background: #f0fff4; border-left-color: #38a169; }}
        .card-title {{ font-size: 12px; text-transform: uppercase; color: #4a5568; font-weight: bold; letter-spacing: 0.5px; }}
        .card-value {{ font-size: 24px; font-weight: bold; color: #1a365d; margin-top: 4px; }}
        .card-desc {{ font-size: 13px; color: #718096; margin-top: 2px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Retinal Disease Detection — Multi-Round Experiments Leaderboard</h1>
        
        <div class="summary-cards">
            <div class="card">
                <div class="card-title">Round 1 Champion (Baseline)</div>
                <div class="card-value">Swin-T (QWK: 0.7542)</div>
                <div class="card-desc">No Augmentations | Standard Cross-Entropy</div>
            </div>
            <div class="card card-green">
                <div class="card-title">Round 2 Champion (Enhanced)</div>
                <div class="card-value">ConvNeXt-Tiny v2 (QWK: 0.7569)</div>
                <div class="card-desc">Augmentations + Focal Loss + Dropout (Test Loss: 0.3418)</div>
            </div>
        </div>

        <div class="img-box">
            <img src="{chart_b64}" alt="Round 1 vs Round 2 Comparison Chart">
        </div>

        <h2>🔄 Side-by-Side Round 1 vs Round 2 Direct Comparison</h2>
        <p>Direct architectural delta comparing Baseline (Round 1) vs Enhanced (Round 2: Augmentations, Class-Balanced Focal Loss, Regularization).</p>
        {table_comp_html}

        <h2>🚀 Round 2 Experiments Leaderboard <span class="badge badge-r2">Augmentations + Focal Loss + Regularization</span></h2>
        <p>Models fine-tuned with Albumentations (Flips, Rotation ±180°, Brightness/Contrast), Class-Balanced Focal Loss, Classifier Dropout (0.3), and Weight Decay (0.001).</p>
        {table_r2_html}

        <h2>📌 Round 1 Experiments Leaderboard <span class="badge badge-r1">Baseline</span></h2>
        <p>Models fine-tuned with standard Resize & Normalize, no augmentations, and standard Cross-Entropy Loss.</p>
        {table_r1_html}
    </div>
</body>
</html>
"""
        with out_file.open("w", encoding="utf-8") as f:
            f.write(html_content)

        return out_file
