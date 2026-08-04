"""Benchmark 5 SOTA XAI algorithms side by side and generate comparison grid."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.visualization.render_xai_slide_table import generate_xai_slide_table


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 5 SOTA XAI algorithms benchmark")
    parser.add_argument(
        "--clf-config",
        type=Path,
        default=Path("configs/experiments/exp_05_convnext_tiny.yaml"),
        help="Classification config path",
    )
    parser.add_argument(
        "--clf-ckpt",
        type=Path,
        default=Path("experiments/exp_05_convnext_tiny_v2/checkpoints/best.pt"),
        help="Classification checkpoint path",
    )
    parser.add_argument(
        "--det-config",
        type=Path,
        default=Path("configs/config.yaml"),
        help="Detection config path",
    )
    parser.add_argument(
        "--det-ckpt",
        type=Path,
        default=Path("experiments/exp_faster_rcnn_lesions_v3_1536/checkpoints/best.pt"),
        help="Detection checkpoint path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/figures/xai_pipeline_comparison_table.png"),
        help="Output image path",
    )

    args, _ = parser.parse_known_args()

    # Search for alternative checkpoints if the defaults are not available
    clf_ckpt = args.clf_ckpt
    if not clf_ckpt.exists():
        for p in Path("experiments").rglob("*.pt"):
            if "lesions" not in str(p).lower():
                clf_ckpt = p
                break

    det_ckpt = args.det_ckpt
    if not det_ckpt.exists():
        for p in Path("experiments").rglob("*.pt"):
            if "lesions" in str(p).lower():
                det_ckpt = p
                break

    print("[BENCHMARK-XAI] Starting XAI benchmark...")
    print(f"  Classification Ckpt: {clf_ckpt}")
    print(f"  Detection Ckpt:      {det_ckpt}")
    print(f"  Output Table Image:  {args.output}")

    generate_xai_slide_table(
        clf_config_path=args.clf_config,
        clf_ckpt_path=clf_ckpt,
        det_config_path=args.det_config,
        det_ckpt_path=det_ckpt,
        output_image_path=args.output,
    )
    print(f"[BENCHMARK-XAI] Completed! Table saved to: {args.output}")


if __name__ == "__main__":
    main()
