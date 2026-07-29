"""Gradient-weighted Class Activation Mapping (Grad-CAM) for Retinal Disease Detection."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch import nn


def _find_target_layer(model: nn.Module) -> nn.Module:
    """Locate the target feature layer for Grad-CAM in a PyTorch model.

    Strategy:
      1. For CNN architectures (ResNet, EfficientNet, DenseNet, MobileNet, ConvNeXt):
         use the last nn.Conv2d layer, which produces spatial feature maps.
      2. For Vision Transformers (ViT, Swin-T) that lack Conv2d in their encoder:
         fall back to the last nn.LayerNorm, which normalizes the spatial token
         representations and still carries gradient information suitable for CAM.
    """
    last_conv = None
    last_norm = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module
        elif isinstance(module, nn.LayerNorm):
            last_norm = module

    if last_conv is not None:
        return last_conv
    if last_norm is not None:
        return last_norm
    raise ValueError(
        "Could not automatically locate a nn.Conv2d or nn.LayerNorm layer in model."
    )


def overlay_heatmap(
    image_rgb: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """Blend a Grad-CAM heatmap with an RGB fundus image.

    Args:
        image_rgb: RGB image numpy array (H, W, 3), dtype uint8 or float [0, 1].
        heatmap: Single channel heatmap numpy array (H, W), values in [0, 1].
        alpha: Blend ratio for heatmap overlay.
        colormap: OpenCV colormap enum (default COLORMAP_JET).

    Returns:
        RGB numpy array (H, W, 3) uint8 of blended visualization.
    """
    if image_rgb.dtype != np.uint8:
        img_uint8 = (np.clip(image_rgb, 0, 1) * 255).astype(np.uint8)
    else:
        img_uint8 = image_rgb.copy()

    heatmap_uint8 = (np.clip(heatmap, 0, 1) * 255).astype(np.uint8)
    colored_heatmap = cv2.applyColorMap(heatmap_uint8, colormap)
    colored_heatmap_rgb = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB)

    blended = cv2.addWeighted(img_uint8, 1 - alpha, colored_heatmap_rgb, alpha, 0)
    return blended


class GradCAM:
    """Gradient-weighted Class Activation Mapping engine."""

    def __init__(self, model: nn.Module, target_layer: nn.Module | None = None) -> None:
        """Initialize Grad-CAM.

        Args:
            model: Neural network model.
            target_layer: Specific layer instance to hook. If None, auto-locates last Conv2d layer.
        """
        self.model = model
        self.target_layer = target_layer if target_layer is not None else _find_target_layer(model)

        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None

        self._register_hooks()

    def _register_hooks(self) -> None:
        def forward_hook(module: nn.Module, input: Any, output: torch.Tensor) -> None:
            self.activations = output.detach()

        def backward_hook(module: nn.Module, grad_input: Any, grad_output: tuple[torch.Tensor, ...]) -> None:
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: int | None = None,
    ) -> np.ndarray:
        """Compute Grad-CAM heatmap for an input image tensor.

        Args:
            input_tensor: Image tensor of shape (1, C, H, W).
            target_class: Target class index. If None, uses top predicted class.

        Returns:
            Normalized 2D numpy array heatmap (H, W) in range [0.0, 1.0].
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

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Grad-CAM hooks failed to capture gradients or activations.")

        gradients = self.gradients[0]  # (C, H_feat, W_feat)
        activations = self.activations[0]  # (C, H_feat, W_feat)

        weights = torch.mean(gradients, dim=(1, 2), keepdim=True)  # (C, 1, 1)
        cam = torch.sum(weights * activations, dim=0)  # (H_feat, W_feat)

        cam = torch.clamp(cam, min=0)  # ReLU
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
        """Generate and save visual heatmap overlay to disk.

        Args:
            input_tensor: Preprocessed tensor (1, C, H, W).
            original_rgb: Original image numpy RGB (H, W, 3).
            save_path: Path to save PNG visualization.
            target_class: Optional class target.
            alpha: Blend factor.

        Returns:
            Path object to saved visualization.
        """
        heatmap = self.generate(input_tensor=input_tensor, target_class=target_class)
        overlay = overlay_heatmap(original_rgb, heatmap, alpha=alpha)

        out_path = Path(save_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out_path), bgr)

        return out_path
