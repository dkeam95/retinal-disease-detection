"""Automated Grad-CAM Batch Audit & Gallery Generator for Retinal Disease Explainability.

Supports two visualization modes:
    ``"heatmap"``   -- classic JET heatmap overlay (original behaviour)
    ``"spotlight"`` -- CLAHE-enhanced contour spotlight (single annotated image)
    ``"4panel"``    -- 2x2 grid: Original | CLAHE | GradCAM | Contour Spotlight
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from explainability.gradcam import GradCAM
from explainability.spotlight import SpotlightVisualizer
from inference.predictor import RetinalPredictor


class GradCAMBatchAuditor:
    """Runs explainability audits by sampling dataset images and generating visualizations.

    Supports three output modes controlled by the ``mode`` parameter in ``run_audit``:
        ``"heatmap"``   -- original blended JET heatmap overlay
        ``"spotlight"`` -- CLAHE-enhanced contour spotlight image
        ``"4panel"``    -- 2x2 panel: Original | CLAHE | GradCAM | Spotlight
    """

    def __init__(
        self,
        predictor: RetinalPredictor,
        output_dir: str | Path = "reports/figures/gradcam_audit",
        spotlight_threshold: float = 0.45,
        panel_size: tuple[int, int] = (320, 320),
    ) -> None:
        """Initialize GradCAMBatchAuditor.

        Args:
            predictor: Instantiated RetinalPredictor.
            output_dir: Root directory for audit outputs.
            spotlight_threshold: Activation threshold for contour detection (0–1).
            panel_size: (width, height) for each panel in 4-panel mode.
        """
        self.predictor = predictor
        self.gradcam = GradCAM(model=predictor.model)
        self.output_dir = Path(output_dir)
        self.visualizer = SpotlightVisualizer(
            threshold=spotlight_threshold,
            panel_size=panel_size,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def run_audit(
        self,
        dataloader: DataLoader | list,
        experiment_name: str = "gradcam_audit",
        max_samples_per_class: int = 3,
        mode: str = "4panel",
    ) -> dict[str, Any]:
        """Run batch Grad-CAM audit over a dataloader, sampling up to N images per class.

        Args:
            dataloader: DataLoader or list of pre-collected BatchSample objects.
            experiment_name: Experiment identifier used as the output subfolder name.
            max_samples_per_class: Maximum images to save per target DR grade.
            mode: Visualization mode — one of ``"heatmap"``, ``"spotlight"``, ``"4panel"``.

        Returns:
            Dictionary with keys:
                ``experiment_name``, ``mode``, ``total_samples``,
                ``gallery_html``, ``samples``.
        """
        exp_dir = self.output_dir / experiment_name
        exp_dir.mkdir(parents=True, exist_ok=True)

        class_counts: dict[int, int] = {
            i: 0 for i in range(len(self.predictor.class_names))
        }
        saved_samples: list[dict[str, Any]] = []

        for b_idx, batch in enumerate(dataloader):
            if b_idx >= 10 or all(
                c >= max_samples_per_class for c in class_counts.values()
            ):
                break

            images, targets, paths = self._unpack_batch(batch)
            if images is None:
                continue

            for idx in range(len(targets)):
                gt_label = int(
                    targets[idx].item()
                    if isinstance(targets[idx], torch.Tensor)
                    else targets[idx]
                )
                if class_counts.get(gt_label, 0) >= max_samples_per_class:
                    continue

                img_path = paths[idx] if paths else None
                image_id = (
                    Path(str(img_path)).name if img_path else f"img_{b_idx}_{idx}.jpg"
                )

                # Load original RGB for visualization
                original_rgb = self._load_original_rgb(img_path, images[idx])

                # Prepare preprocessed tensor for GradCAM
                tensor = images[idx].unsqueeze(0).to(self.predictor.device)

                # Forward pass: get prediction
                outputs = self.predictor.model(tensor)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs
                probs = torch.softmax(logits, dim=1).squeeze(0).detach().cpu().numpy()
                pred_label = int(np.argmax(probs))
                confidence = float(probs[pred_label])

                # Generate Grad-CAM heatmap
                heatmap = self.gradcam.generate(
                    input_tensor=tensor, target_class=pred_label
                )

                # Build output filename
                out_path = self._build_output_path(
                    exp_dir=exp_dir,
                    image_id=image_id,
                    experiment_name=experiment_name,
                    gt_label=gt_label,
                    pred_label=pred_label,
                    confidence=confidence,
                    mode=mode,
                )

                # Render + save visualization
                class_name = self.predictor.class_names[pred_label]
                self.visualizer.save(
                    original_rgb=original_rgb,
                    heatmap=heatmap,
                    save_path=out_path,
                    grade_id=pred_label,
                    class_name=class_name,
                    confidence=confidence,
                    image_id=Path(image_id).stem,
                    mode=mode,
                )

                saved_samples.append(
                    {
                        "image_id": image_id,
                        "true_label": gt_label,
                        "true_name": self.predictor.class_names[gt_label],
                        "pred_label": pred_label,
                        "pred_name": class_name,
                        "confidence": confidence,
                        "cam_path": str(out_path),
                        "mode": mode,
                    }
                )
                class_counts[gt_label] = class_counts.get(gt_label, 0) + 1

            if all(count >= max_samples_per_class for count in class_counts.values()):
                break

        html_gallery = self.export_html_gallery(
            saved_samples, experiment_name=experiment_name, mode=mode
        )
        return {
            "experiment_name": experiment_name,
            "mode": mode,
            "total_samples": len(saved_samples),
            "gallery_html": str(html_gallery),
            "samples": saved_samples,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # HTML Gallery
    # ─────────────────────────────────────────────────────────────────────────

    def export_html_gallery(
        self,
        samples: list[dict[str, Any]],
        experiment_name: str = "gradcam_audit",
        mode: str = "4panel",
    ) -> Path:
        """Export an HTML visual gallery displaying the generated visualizations."""
        out_file = self.output_dir / experiment_name / "gradcam_gallery.html"
        out_file.parent.mkdir(parents=True, exist_ok=True)

        mode_label = {
            "heatmap": "Grad-CAM Heatmap Overlay",
            "spotlight": "CLAHE Contour Spotlight",
            "4panel": "CLAHE + Spotlight 4-Panel Report",
        }.get(mode, mode)

        cards_html = []
        for s in samples:
            rel_path = Path(s["cam_path"]).name
            is_correct = s["true_label"] == s["pred_label"]
            status_color = "#28a745" if is_correct else "#dc3545"
            status_icon = "V" if is_correct else "X"
            cards_html.append(f"""
            <div class="card">
                <img src="{html.escape(rel_path)}" alt="{html.escape(str(s["image_id"]))}">
                <div class="card-body">
                    <div class="card-title">{html.escape(str(s["image_id"]))}</div>
                    <div class="info"><strong>GT:</strong> Grade {s["true_label"]} ({html.escape(str(s["true_name"]))})</div>
                    <div class="info" style="color: {status_color};">
                        <strong>{status_icon} Pred:</strong> Grade {s["pred_label"]}
                        ({html.escape(str(s["pred_name"]))}) [{s["confidence"]:.1%}]
                    </div>
                </div>
            </div>
            """)

        gallery_html = "".join(cards_html)
        card_width = "660px" if mode == "4panel" else "360px"

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Explainability Audit - {experiment_name}</title>
    <style>
        body {{ font-family: 'Segoe UI', sans-serif; background: #0f1117; padding: 25px; color: #e2e8f0; }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        h1 {{ color: #63b3ed; border-bottom: 2px solid #2d3748; padding-bottom: 10px; }}
        .meta {{ color: #a0aec0; font-size: 13px; margin-bottom: 20px; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax({card_width}, 1fr)); gap: 20px; margin-top: 25px; }}
        .card {{ background: #1a202c; border: 1px solid #2d3748; border-radius: 10px; overflow: hidden; }}
        .card img {{ width: 100%; height: auto; display: block; }}
        .card-body {{ padding: 12px 15px; }}
        .card-title {{ font-weight: bold; font-size: 13px; margin-bottom: 6px; color: #90cdf4; word-break: break-all; }}
        .info {{ font-size: 12px; margin-top: 4px; color: #cbd5e0; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Explainability Audit &mdash; {experiment_name}</h1>
        <p class="meta">Mode: <strong>{mode_label}</strong> &nbsp;|&nbsp;
           Samples: <strong>{len(samples)}</strong></p>
        <div class="grid">
            {gallery_html}
        </div>
    </div>
</body>
</html>
"""
        with out_file.open("w", encoding="utf-8") as f:
            f.write(html_content)

        return out_file

    # ─────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _unpack_batch(batch: Any) -> tuple[Any | None, Any | None, list | None]:
        """Unpack a batch into (images, targets, paths) regardless of batch type."""
        if hasattr(batch, "image") and hasattr(batch, "label"):
            images = batch.image
            targets = batch.label
            # BatchSample uses image_paths (plural)
            paths = list(
                getattr(batch, "image_paths", [])
                or getattr(batch, "image_path", [])
                or []
            )
        elif isinstance(batch, (list, tuple)):
            images = batch[0]
            targets = batch[1]
            paths = list(batch[2]) if len(batch) > 2 else []
        elif isinstance(batch, dict):
            images = batch["image"]
            targets = batch["label"]
            paths = list(batch.get("path", batch.get("paths", [])))
        else:
            return None, None, None
        return images, targets, paths

    @staticmethod
    def _load_original_rgb(img_path: Any, tensor: torch.Tensor) -> np.ndarray:
        """Load original RGB image from disk or reconstruct from a normalised tensor."""
        if img_path is not None:
            p = Path(str(img_path))
            if p.exists():
                bgr = cv2.imread(str(p))
                if bgr is not None:
                    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        # Fallback: reverse ImageNet normalisation from the preprocessed tensor
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_np = tensor.cpu().numpy().transpose(1, 2, 0)  # (H, W, C)
        img_np = img_np * std + mean
        return (np.clip(img_np, 0, 1) * 255).astype(np.uint8)

    @staticmethod
    def _build_output_path(
        exp_dir: Path,
        image_id: str,
        experiment_name: str,
        gt_label: int,
        pred_label: int,
        confidence: float,
        mode: str,
    ) -> Path:
        stem = Path(image_id).stem
        fname = f"{experiment_name}_{stem}_gt{gt_label}_pred{pred_label}_conf{confidence:.2f}_{mode}.png"
        return exp_dir / fname
