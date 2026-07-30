"""Unified Explainability Engine for Retinal Disease Classification.

Provides a single, standardized interface for generating feature attribution heatmaps
and visual explanations using any of the 5 supported XAI algorithms:
  - 'gradcam': Baseline Grad-CAM
  - 'gradcam++': Grad-CAM++ (higher-order gradient weighting for multiple lesions)
  - 'layer_cam': Layer-CAM (fine-grained spatial gradient preservation)
  - 'score_cam': Score-CAM (gradient-free perturbation confidence scoring)
  - 'integrated_gradients': Integrated Gradients (pixel-level axiomatic attribution)

Includes automated ROI Retina Masking to zero-out uninformative outer black backgrounds
and eliminate edge-contrast artifacts around the retina circle.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

from explainability.gradcam import GradCAM
from explainability.gradcam_plus_plus import GradCAMPlusPlus
from explainability.integrated_gradients import IntegratedGradients
from explainability.layer_cam import LayerCAM
from explainability.score_cam import ScoreCAM
from explainability.spotlight import SpotlightVisualizer
from preprocessing.roi import get_retina_binary_mask


class ExplainabilityEngine:
    """Unified engine encapsulating all 5 XAI algorithms with ROI Retina Masking."""

    SUPPORTED_ALGORITHMS = (
        "gradcam",
        "gradcam++",
        "layer_cam",
        "score_cam",
        "integrated_gradients",
    )

    def __init__(
        self,
        model: nn.Module,
        target_layer: nn.Module | None = None,
        enable_roi_masking: bool = True,
    ) -> None:
        """Initialize Explainability Engine.

        Args:
            model: PyTorch neural network model.
            target_layer: Optional target layer for CAM-based algorithms.
            enable_roi_masking: Whether to apply automated Retina ROI Masking to zero out background noise.
        """
        self.model = model
        self.target_layer = target_layer
        self.enable_roi_masking = enable_roi_masking

        # Instantiated XAI algorithm backends
        self._gradcam = GradCAM(model=model, target_layer=target_layer)
        self._gradcam_pp = GradCAMPlusPlus(model=model, target_layer=target_layer)
        self._layer_cam = LayerCAM(model=model, target_layers=target_layer)
        self._score_cam = ScoreCAM(model=model, target_layer=target_layer)
        self._ig = IntegratedGradients(model=model, n_steps=30)

        self.spotlight_vis = SpotlightVisualizer()

    def generate_heatmap(
        self,
        input_tensor: torch.Tensor,
        algorithm: str = "gradcam++",
        target_class: int | None = None,
        original_rgb: np.ndarray | None = None,
    ) -> np.ndarray:
        """Generate normalized 2D heatmap [0.0, 1.0] using the chosen XAI algorithm.

        Args:
            input_tensor: Preprocessed tensor (1, C, H, W).
            algorithm: One of 'gradcam', 'gradcam++', 'layer_cam', 'score_cam', 'integrated_gradients'.
            target_class: Target class index. If None, top prediction is used.
            original_rgb: Optional original RGB image for ROI masking calculation.

        Returns:
            Normalised 2D numpy array heatmap (H, W) in range [0.0, 1.0].
        """
        algo = algorithm.lower()
        if algo not in self.SUPPORTED_ALGORITHMS:
            raise ValueError(
                f"Unknown algorithm '{algorithm}'. Choose from: {self.SUPPORTED_ALGORITHMS}"
            )

        if algo == "gradcam":
            heatmap = self._gradcam.generate(input_tensor, target_class=target_class)
        elif algo in ("gradcam++", "gradcam_pp"):
            heatmap = self._gradcam_pp.generate(input_tensor, target_class=target_class)
        elif algo == "layer_cam":
            heatmap = self._layer_cam.generate(input_tensor, target_class=target_class)
        elif algo == "score_cam":
            heatmap = self._score_cam.generate(input_tensor, target_class=target_class)
        elif algo == "integrated_gradients":
            heatmap = self._ig.generate(input_tensor, target_class=target_class)
        else:
            raise RuntimeError(f"Failed to route algorithm '{algorithm}'")

        # Apply ROI Retina Masking if enabled and original_rgb provided (or reconstructed from tensor)
        if self.enable_roi_masking:
            if original_rgb is None:
                # Reconstruct un-normalized RGB image from tensor for mask extraction
                mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
                std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
                img_np = (
                    input_tensor.squeeze(0).cpu().detach().numpy().transpose(1, 2, 0)
                )
                original_rgb = (np.clip(img_np * std + mean, 0, 1) * 255).astype(
                    np.uint8
                )

            mask = get_retina_binary_mask(original_rgb)
            if mask.shape != heatmap.shape:
                mask = cv2.resize(mask, (heatmap.shape[1], heatmap.shape[0]))

            # Multiply heatmap by binary retina mask (zeros out outer black background)
            heatmap = heatmap * mask
            if heatmap.max() > 0:
                heatmap = heatmap / heatmap.max()

        return heatmap

    def save_visualization(
        self,
        input_tensor: torch.Tensor,
        original_rgb: np.ndarray,
        save_path: str | Path,
        algorithm: str = "gradcam++",
        target_class: int | None = None,
        grade_id: int = 2,
        class_name: str = "Moderate NPDR",
        confidence: float = 0.0,
        image_id: str = "",
        mode: str = "4panel",
    ) -> Path:
        """Generate and save XAI visualization using the chosen algorithm and visual mode."""
        heatmap = self.generate_heatmap(
            input_tensor=input_tensor,
            algorithm=algorithm,
            target_class=target_class,
            original_rgb=original_rgb,
        )

        return self.spotlight_vis.save(
            original_rgb=original_rgb,
            heatmap=heatmap,
            save_path=save_path,
            grade_id=grade_id,
            class_name=class_name,
            confidence=confidence,
            image_id=image_id,
            mode=mode,
        )
