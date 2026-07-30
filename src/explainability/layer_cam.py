"""Layer-CAM (Layer-wise Class Activation Mapping) implementation.

Paper: Layer-CAM: Exploring Hierarchical Class Activation Maps for Localization
       Jiang et al., 2021 (https://ieeexplore.ieee.org/document/9462463)

Layer-CAM produces fine-grained, high-resolution class activation maps by combining element-wise
positive gradients with activation maps across single or multiple layers of a neural network.
Unlike Grad-CAM which applies global average pooling to gradients, Layer-CAM preserves spatial
variations at each location, making it exceptionally suited for identifying small anatomical structures,
vessels, and micro-lesions in retinal fundus photography.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch import nn

from explainability.gradcam import _find_target_layer, overlay_heatmap


class LayerCAM:
    """Layer-CAM explainability engine for fine-grained, multi-layer activation mapping."""

    def __init__(
        self,
        model: nn.Module,
        target_layers: nn.Module | Sequence[nn.Module] | None = None,
    ) -> None:
        """Initialize Layer-CAM.

        Args:
            model: PyTorch neural network model.
            target_layers: Target layer or sequence of target layers. If None, auto-locates last Conv2d layer.
        """
        self.model = model

        if target_layers is None:
            self.target_layers = [_find_target_layer(model)]
        elif isinstance(target_layers, nn.Module):
            self.target_layers = [target_layers]
        else:
            self.target_layers = list(target_layers)

        self.activations: dict[nn.Module, torch.Tensor] = {}
        self.gradients: dict[nn.Module, torch.Tensor] = {}

        self._register_hooks()

    def _register_hooks(self) -> None:
        for layer in self.target_layers:

            def make_forward_hook(target_layer: nn.Module):
                def hook(module: nn.Module, input: Any, output: torch.Tensor) -> None:
                    self.activations[target_layer] = output.detach()

                return hook

            def make_backward_hook(target_layer: nn.Module):
                def hook(
                    module: nn.Module,
                    grad_input: Any,
                    grad_output: tuple[torch.Tensor, ...],
                ) -> None:
                    self.gradients[target_layer] = grad_output[0].detach()

                return hook

            layer.register_forward_hook(make_forward_hook(layer))
            layer.register_full_backward_hook(make_backward_hook(layer))

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: int | None = None,
    ) -> np.ndarray:
        """Generate fine-grained Layer-CAM heatmap for an input image tensor.

        Args:
            input_tensor: Preprocessed tensor (1, C, H, W).
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
        score.backward()

        h, w = input_tensor.shape[2], input_tensor.shape[3]
        combined_cam = np.zeros((h, w), dtype=np.float32)

        for layer in self.target_layers:
            if layer not in self.gradients or layer not in self.activations:
                continue

            grads = self.gradients[layer][0]  # (C, H_feat, W_feat)
            acts = self.activations[layer][0]  # (C, H_feat, W_feat)

            if grads.ndim == 2:
                spatial_dim = int(np.sqrt(grads.shape[0]))
                grads = grads.permute(1, 0).view(-1, spatial_dim, spatial_dim)
                acts = acts.permute(1, 0).view(-1, spatial_dim, spatial_dim)

            # Element-wise positive gradients * activations (no spatial pooling!)
            pos_grads = torch.clamp(grads, min=0.0)
            layer_map = pos_grads * acts

            # Sum across channels
            cam_layer = torch.sum(layer_map, dim=0)
            cam_layer = torch.clamp(cam_layer, min=0.0)
            cam_np = cam_layer.cpu().numpy()

            if cam_np.max() > 0:
                cam_np = cam_np / cam_np.max()

            cam_resized = cv2.resize(cam_np, (w, h))
            combined_cam += cam_resized

        if combined_cam.max() > 0:
            combined_cam = combined_cam / combined_cam.max()

        return combined_cam

    def save_visualization(
        self,
        input_tensor: torch.Tensor,
        original_rgb: np.ndarray,
        save_path: str | Path,
        target_class: int | None = None,
        alpha: float = 0.5,
    ) -> Path:
        """Generate and save fine-grained Layer-CAM visualization overlay to disk."""
        heatmap = self.generate(input_tensor=input_tensor, target_class=target_class)
        overlay = overlay_heatmap(original_rgb, heatmap, alpha=alpha)

        out_path = Path(save_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out_path), bgr)
        return out_path
