"""Master CLI Entrypoint for Retinal Disease Detection & AI Analytics System.

Usage:
    python main.py                        # Run complete master clinical diagnostic & AI analytics pipeline
    python main.py --mode master-report   # Generate master clinical HTML diagnostic report
    python main.py --mode demo            # Run live classification ensemble & error analysis demo
    python main.py --mode benchmark-xai   # Benchmark 5 SOTA XAI algorithms
    python main.py --mode benchmark-det   # Run 3-round Faster R-CNN detection resolution benchmark
    python main.py --mode train-detection # Train Faster R-CNN lesion detector
    python main.py --mode pdf-report       # Generate publication-grade PDF report
"""

from __future__ import annotations

import argparse
import os
import sys

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

from pathlib import Path

# Ensure src/ is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Master Entrypoint for Retinal Disease Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                         Run complete master clinical pipeline
  python main.py --mode benchmark-det    Run 3-round detection resolution benchmark
  python main.py --mode benchmark-xai    Run 5 SOTA XAI algorithms benchmark
        """,
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="master-report",
        choices=[
            "master-report",
            "demo",
            "benchmark-xai",
            "benchmark-seg",
            "benchmark-det",
            "train-detection",
            "pdf-report",
            "all",
        ],
        help="Pipeline mode to execute (default: master-report)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to YAML configuration file (used for train-detection)",
    )
    parser.add_argument(
        "--target-grades",
        type=str,
        default=None,
        help="Comma-separated target DR grades to test, e.g. '3,4' or '0,1,2,3,4'",
    )
    parser.add_argument(
        "--images",
        type=str,
        default=None,
        help="Comma-separated image stems or filenames to test, e.g. '007-1774-100,007-3396-200'",
    )
    args = parser.parse_args()

    mode = args.mode

    if mode in ("master-report", "all"):
        print("[INIT] Executing Master Clinical Diagnostic & AI Analytics Pipeline...\n")
        # Ensure Presentation Comparison Grid Tables are generated
        try:
            from scripts.visualization.render_segmentation_slide_table import generate_segmentation_slide_table
            from scripts.visualization.render_xai_slide_table import generate_xai_slide_table
            generate_segmentation_slide_table()
            generate_xai_slide_table()
        except Exception as e:
            print(f"[WARNING] Could not auto-render presentation tables: {e}")

        from scripts.reporting.generate_master_clinical_reports import main as run_master

        t_grades = [int(g.strip()) for g in args.target_grades.split(",") if g.strip().isdigit()] if args.target_grades else None
        c_images = [img.strip() for img in args.images.split(",") if img.strip()] if args.images else None

        run_master(target_grades=t_grades, custom_images=c_images)

    elif mode == "demo":
        print("[INIT] Executing Live Pipeline Demo (Ensemble + Error Analysis + Spotlight)...\n")
        from scripts.reporting.run_demo_pipeline import main as run_demo

        run_demo()

    elif mode == "benchmark-xai":
        print("[INIT] Executing 5 SOTA XAI Algorithms Side-by-Side Benchmark & Grid Table...\n")
        from scripts.evaluation.benchmark_xai import main as run_xai

        run_xai()

    elif mode == "benchmark-seg":
        print("[INIT] Executing 5-Column Lesion Segmentation Pipeline Grid Renderer (Slide 3)...\n")
        from scripts.visualization.render_segmentation_slide_table import generate_segmentation_slide_table

        generate_segmentation_slide_table()

    elif mode == "benchmark-det":
        print("[INIT] Executing 3-Round Faster R-CNN Detection Resolution Benchmark...\n")
        from scripts.evaluation.compare_detection_rounds import main as run_det_bench

        run_det_bench()

    elif mode == "train-detection":
        print(f"[INIT] Launching Faster R-CNN Lesion Detector Training with config: {args.config}...\n")
        from scripts.training.train_detection import main as run_train_det

        sys.argv = ["train_detection.py", "--config", args.config]
        run_train_det()

    elif mode == "pdf-report":
        print("[INIT] Generating Publication-Grade PDF Executive Report...\n")
        from scripts.reporting.export_pdf_reports import main as export_pdfs

        export_pdfs()
        from scripts.reporting.generate_pdf_report import build_pdf_report

        build_pdf_report()


if __name__ == "__main__":
    main()
