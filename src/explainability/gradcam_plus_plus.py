"""Grad-CAM++ (Gradient-weighted Class Activation Mapping Plus Plus) implementation.

Paper: Grad-CAM++: Generalized Gradient-Based Visual Explanations for Deep Convolutional Networks
       Chattopadhay et al., 2018 (https://arxiv.org/abs/1710.11063)

Grad-CAM++ uses second and third-order gradients of target logits with respect to feature map
activations to weight the feature maps. Unlike standard Grad-CAM which averages gradients globally,
Grad-CAM++ calculates pixel-wise weighting coefficients, allowing it to capture multiple distinct
lesions/pathologies occurring in the same fundus image.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch import nn

from explainability.gradcam import _find_target_layer, overlay_heatmap


class GradCAMPlusPlus:
    """Grad-CAM++ explainability engine for Convolutional & Transformer architectures."""

    def __init__(self, model: nn.Module, target_layer: nn.Module | None = None) -> None:
        """Initialize Grad-CAM++.

        Args:
            model: Neural network model (PyTorch nn.Module).
            target_layer: Target feature layer. If None, auto-locates last Conv2d/LayerNorm layer.
        """
        self.model = model
        self.target_layer = (
            target_layer if target_layer is not None else _find_target_layer(model)
        )

        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None

        self._register_hooks()

    def _register_hooks(self) -> None:
        def forward_hook(module: nn.Module, input: Any, output: torch.Tensor) -> None:
            self.activations = output.detach()

        def backward_hook(
            module: nn.Module, grad_input: Any, grad_output: tuple[torch.Tensor, ...]
        ) -> None:
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: int | None = None,
    ) -> np.ndarray:
        """Generate Grad-CAM++ heatmap for an input image tensor.

        Args:
            input_tensor: Preprocessed image tensor of shape (1, C, H, W).
            target_class: Target class index. If None, uses top predicted class.

        Returns:
            Normalised 2D numpy array heatmap (H, W) in range [0.0, 1.0].
        """
        self.model.eval()
        self.model.zero_grad()

        input_tensor = input_tensor.requires_grad_(True)
        outputs = self.model(input_tensor)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs

        if target_class is None:
            target_class = int(torch.argmax(logits, dim=1).item())

        score = logits[0, target_class]
        # Exponentiate score for positive higher-order derivative scaling
        score_exp = torch.exp(score)
        score_exp.backward()

        if self.gradients is None or self.activations is None:
            raise RuntimeError(
                "Grad-CAM++ hooks failed to capture gradients or activations."
            )

        # Shapes:
        # gradients:   (1, C, H_feat, W_feat)
        # activations: (1, C, H_feat, W_feat)
        grads = self.gradients[0]  # (C, H_feat, W_feat)
        acts = self.activations[0]  # (C, H_feat, W_feat)

        # Handle 2D transformer activations (1, N, C) -> reshape to 3D if needed
        if grads.ndim == 2:
            # (N, C) -> (C, N_sqrt, N_sqrt)
            spatial_dim = int(np.sqrt(grads.shape[0]))
            grads = grads.permute(1, 0).view(-1, spatial_dim, spatial_dim)
            acts = acts.permute(1, 0).view(-1, spatial_dim, spatial_dim)

        # Compute higher order gradients
        grads_pow2 = grads**2
        grads_pow3 = grads**3

        sum_acts = torch.sum(acts, dim=(1, 2), keepdim=True)  # (C, 1, 1)
        eps = 1e-8

        # Alpha weighting denominator: 2 * grad^2 + sum(A) * grad^3
        alpha_denom = 2.0 * grads_pow2 + sum_acts * grads_pow3 + eps
        alpha = grads_pow2 / alpha_denom

        # ReLU on first gradients (only positive contribution to target class)
        positive_grads = torch.clamp(grads, min=0.0)

        # Class weights w_k = sum_ij (alpha_ij * relu(grad_ij))
        weights = torch.sum(
            alpha * positive_grads, dim=(1, 2), keepdim=True
        )  # (C, 1, 1)

        # Weighted combination of activation maps
        cam = torch.sum(weights * acts, dim=0)  # (H_feat, W_feat)
        cam = torch.clamp(cam, min=0.0)

        cam_np = cam.cpu().numpy()
        if cam_np.max() > 0:
            cam_np = cam_np / cam_np.max()

        h, w = input_tensor.shape[2], input_tensor.shape[3]
        heatmap_resized = cv2.resize(cam_np, (w, h))
        return heatmap_resized

    def save_visualization(
        self,
        input_tensor: torch.Tensor,
        original_rgb: np.ndarray,
        save_path: str | Path,
        target_class: int | None = None,
        alpha: float = 0.5,
    ) -> Path:
        """Generate and save Grad-CAM++ visualization overlay to disk."""
        heatmap = self.generate(input_tensor=input_tensor, target_class=target_class)
        overlay = overlay_heatmap(original_rgb, heatmap, alpha=alpha)

        out_path = Path(save_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out_path), bgr)
        return out_path
