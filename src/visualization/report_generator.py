"""HTML Report Generator for Retinal Disease Detection model evaluation and experiments."""

from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from .plots import plot_confusion_matrix, plot_learning_curves, plot_per_class_metrics


def figure_to_base64(fig: plt.Figure) -> str:
    """Convert a Matplotlib figure into a base64 encoded PNG data URI string."""
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return f"data:image/png;base64,{img_b64}"


class HTMLReportGenerator:
    """Generates self-contained HTML reports containing visual graphs and metric breakdowns."""

    @staticmethod
    def generate(
        experiment_name: str,
        metrics: dict[str, Any],
        config_summary: dict[str, Any] | None = None,
        history: dict[str, list[float]] | None = None,
        confusion_matrix: Any | None = None,
        per_class_metrics: dict[str, dict[str, float]] | None = None,
        output_path: str | Path = "reports/metrics/experiment_report.html",
    ) -> Path:
        """Generate and save an HTML evaluation report.

        Args:
            experiment_name: Name of the experiment.
            metrics: Dictionary of main metrics (e.g. {'qwk': 0.85, 'accuracy': 0.88, 'macro_f1': 0.82}).
            config_summary: Key-value configuration overview.
            history: Training history dict for plotting curves.
            confusion_matrix: Confusion matrix array.
            per_class_metrics: Breakdown metrics for each DR grade.
            output_path: Output file path for HTML report.

        Returns:
            Path to the saved HTML file.
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # 1. Render plots to base64 URIs
        curves_b64 = ""
        if history and len(history) > 0:
            fig_curves = plot_learning_curves(
                history, title=f"Training & Validation History - {experiment_name}"
            )
            curves_b64 = figure_to_base64(fig_curves)

        cm_b64 = ""
        if confusion_matrix is not None:
            fig_cm = plot_confusion_matrix(
                confusion_matrix, title="Test Set Confusion Matrix"
            )
            cm_b64 = figure_to_base64(fig_cm)

        per_class_b64 = ""
        if per_class_metrics is not None:
            fig_pc = plot_per_class_metrics(
                per_class_metrics, title="Per-Class Performance Metrics"
            )
            per_class_b64 = figure_to_base64(fig_pc)

        # 2. Build HTML Content
        qwk_val = metrics.get(
            "qwk", metrics.get("val_qwk", metrics.get("test_qwk", "N/A"))
        )
        acc_val = metrics.get(
            "accuracy", metrics.get("val_accuracy", metrics.get("test_accuracy", "N/A"))
        )
        f1_val = metrics.get(
            "macro_f1", metrics.get("val_macro_f1", metrics.get("test_macro_f1", "N/A"))
        )

        def format_metric(val: Any) -> str:
            return f"{float(val):.4f}" if isinstance(val, (float, int)) else str(val)

        config_html = ""
        if config_summary:
            config_rows = "".join(
                f"<tr><td class='cfg-key'>{k}</td><td class='cfg-val'>{v}</td></tr>"
                for k, v in config_summary.items()
            )
            config_html = f"""
            <div class="section">
                <h2>⚙️ Model & Training Configuration</h2>
                <table class="styled-table config-table">
                    <thead><tr><th>Parameter</th><th>Value</th></tr></thead>
                    <tbody>{config_rows}</tbody>
                </table>
            </div>
            """

        per_class_table_html = ""
        if per_class_metrics:
            rows = ""
            for cls_name, cls_m in per_class_metrics.items():
                p = cls_m.get("precision", 0.0)
                r = cls_m.get("recall", 0.0)
                f = cls_m.get("f1", cls_m.get("f1_score", 0.0))
                rows += f"<tr><td><b>{cls_name}</b></td><td>{p:.4f}</td><td>{r:.4f}</td><td>{f:.4f}</td></tr>"
            per_class_table_html = f"""
            <div class="section">
                <h2>📋 Per-Class Performance Breakdown</h2>
                <table class="styled-table">
                    <thead><tr><th>Class / DR Grade</th><th>Precision</th><th>Recall</th><th>F1-Score</th></tr></thead>
                    <tbody>{rows}</tbody>
                </table>
            </div>
            """

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Retinal Disease Detection - {experiment_name}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #f4f7f9;
            color: #333;
            margin: 0;
            padding: 20px;
        }}
        .container {{
            max-width: 1100px;
            margin: 0 auto;
            background: #ffffff;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        }}
        header {{
            border-bottom: 3px solid #2b5c8f;
            padding-bottom: 15px;
            margin-bottom: 25px;
        }}
        h1 {{
            color: #1f3a60;
            margin: 0 0 8px 0;
            font-size: 26px;
        }}
        .subtitle {{
            color: #666;
            font-size: 14px;
        }}
        .metric-cards {{
            display: flex;
            gap: 20px;
            margin-bottom: 30px;
        }}
        .card {{
            flex: 1;
            background: linear-gradient(135deg, #1f3a60, #2b5c8f);
            color: #ffffff;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            box-shadow: 0 2px 8px rgba(0,0,0,0.12);
        }}
        .card.qwk {{ background: linear-gradient(135deg, #1b5e20, #2e7d32); }}
        .card.acc {{ background: linear-gradient(135deg, #0d47a1, #1565c0); }}
        .card.f1 {{ background: linear-gradient(135deg, #4a148c, #6a1b9a); }}
        .card-title {{ font-size: 13px; text-transform: uppercase; letter-spacing: 1px; opacity: 0.9; }}
        .card-value {{ font-size: 32px; font-weight: bold; margin-top: 5px; }}
        .section {{ margin-bottom: 35px; }}
        h2 {{ color: #2b5c8f; font-size: 18px; border-bottom: 1px solid #e0e0e0; padding-bottom: 8px; margin-bottom: 15px; }}
        .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
        .img-box {{ text-align: center; background: #fafafa; border: 1px solid #eee; padding: 10px; border-radius: 6px; }}
        .img-box img {{ max-width: 100%; height: auto; border-radius: 4px; }}
        .styled-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
        }}
        .styled-table th, .styled-table td {{
            padding: 10px 14px;
            text-align: left;
            border-bottom: 1px solid #e0e0e0;
        }}
        .styled-table th {{ background-color: #f0f4f8; color: #1f3a60; font-weight: 600; }}
        .styled-table tr:hover {{ background-color: #f9fbfd; }}
        .cfg-key {{ font-weight: 600; color: #444; width: 40%; }}
        .cfg-val {{ color: #1f3a60; }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🩺 Retinal Disease Detection Evaluation Report</h1>
            <div class="subtitle">Experiment Identifier: <b>{experiment_name}</b> | Generated automatically</div>
        </header>

        <div class="metric-cards">
            <div class="card qwk">
                <div class="card-title">Quadratic Weighted Kappa (QWK)</div>
                <div class="card-value">{format_metric(qwk_val)}</div>
            </div>
            <div class="card acc">
                <div class="card-title">Accuracy</div>
                <div class="card-value">{format_metric(acc_val)}</div>
            </div>
            <div class="card f1">
                <div class="card-title">Macro F1-Score</div>
                <div class="card-value">{format_metric(f1_val)}</div>
            </div>
        </div>

        {config_html}

        {f'<div class="section"><h2>📈 Learning & Performance Trajectory</h2><div class="img-box"><img src="{curves_b64}" alt="Learning Curves"></div></div>' if curves_b64 else ""}

        <div class="grid-2">
            {f'<div class="img-box"><h2>🔲 Confusion Matrix Heatmap</h2><img src="{cm_b64}" alt="Confusion Matrix"></div>' if cm_b64 else ""}
            {f'<div class="img-box"><h2>📊 Per-Class Performance Bar Chart</h2><img src="{per_class_b64}" alt="Per Class Metrics"></div>' if per_class_b64 else ""}
        </div>

        {per_class_table_html}
    </div>
</body>
</html>
"""
        with output_file.open("w", encoding="utf-8") as f:
            f.write(html_content)

        return output_file
