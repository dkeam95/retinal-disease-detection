"""Score-CAM (Score-Weighted Visual Explanations) implementation.

Paper: Score-CAM: Score-Weighted Visual Explanations for Convolutional Neural Networks
       Wang et al., 2020 (https://arxiv.org/abs/1910.01279)

Score-CAM is a gradient-free explainability method. Instead of relying on gradients which can be
noisy or suffer from saturation in deep networks, Score-CAM masks the input image with normalised
feature activation maps and passes the masked inputs through the network. The weight of each feature
channel is directly determined by the increase in confidence (score) for the target class.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch import nn

from explainability.gradcam import _find_target_layer, overlay_heatmap


class ScoreCAM:
    """Gradient-free Score-CAM explainability engine."""

    def __init__(
        self,
        model: nn.Module,
        target_layer: nn.Module | None = None,
        max_channels: int = 64,
        batch_size: int = 16,
    ) -> None:
        """Initialize Score-CAM.

        Args:
            model: PyTorch neural network model.
            target_layer: Target feature layer. If None, auto-locates last Conv2d/LayerNorm layer.
            max_channels: Maximum number of feature channels to evaluate (for speed optimization).
            batch_size: Batch size for forward pass over masked images.
        """
        self.model = model
        self.target_layer = (
            target_layer if target_layer is not None else _find_target_layer(model)
        )
        self.max_channels = max_channels
        self.batch_size = batch_size

        self.activations: torch.Tensor | None = None
        self._register_hook()

    def _register_hook(self) -> None:
        def forward_hook(module: nn.Module, input: Any, output: torch.Tensor) -> None:
            self.activations = output.detach()

        self.target_layer.register_forward_hook(forward_hook)

    @torch.no_grad()
    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: int | None = None,
    ) -> np.ndarray:
        """Generate gradient-free Score-CAM heatmap for an input image tensor.

        Args:
            input_tensor: Preprocessed image tensor of shape (1, C, H, W).
            target_class: Target class index. If None, uses top predicted class.

        Returns:
            Normalised 2D numpy array heatmap (H, W) in range [0.0, 1.0].
        """
        self.model.eval()
        device = input_tensor.device

        # Step 1: Forward pass to extract feature activations & baseline output
        outputs = self.model(input_tensor)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs

        if target_class is None:
            target_class = int(torch.argmax(logits, dim=1).item())

        if self.activations is None:
            raise RuntimeError(
                "Score-CAM hook failed to capture target layer activations."
            )

        acts = self.activations[0]  # (C, H_feat, W_feat)
        if acts.ndim == 2:
            spatial_dim = int(np.sqrt(acts.shape[0]))
            acts = acts.permute(1, 0).view(-1, spatial_dim, spatial_dim)

        num_channels, h_feat, w_feat = acts.shape
        h_in, w_in = input_tensor.shape[2], input_tensor.shape[3]

        # Select top channels with highest variance if channels exceed max_channels
        if num_channels > self.max_channels:
            variances = torch.var(acts, dim=(1, 2))
            top_indices = torch.topk(variances, k=self.max_channels).indices
            acts = acts[top_indices]
            num_channels = self.max_channels

        # Step 2: Normalise activation maps into [0, 1] masks and upsample
        masks = []
        for i in range(num_channels):
            m = acts[i]
            m_min, m_max = m.min(), m.max()
            if m_max > m_min:
                m_norm = (m - m_min) / (m_max - m_min)
            else:
                m_norm = torch.zeros_like(m)

            m_np = m_norm.cpu().numpy()
            m_resized = cv2.resize(m_np, (w_in, h_in))
            masks.append(
                torch.from_numpy(m_resized).to(device=device, dtype=input_tensor.dtype)
            )

        masks_tensor = torch.stack(masks, dim=0).unsqueeze(1)  # (N, 1, H_in, W_in)

        # Step 3: Mask input image and compute class target probabilities
        scores = []
        for i in range(0, num_channels, self.batch_size):
            batch_masks = masks_tensor[i : i + self.batch_size]
            masked_inputs = input_tensor * batch_masks

            batch_outputs = self.model(masked_inputs)
            batch_logits = (
                batch_outputs.logits
                if hasattr(batch_outputs, "logits")
                else batch_outputs
            )
            batch_probs = torch.softmax(batch_logits, dim=1)[:, target_class]
            scores.append(batch_probs)

        score_weights = torch.cat(scores, dim=0)  # (N,)
        score_weights = torch.softmax(score_weights, dim=0).view(-1, 1, 1)  # (N, 1, 1)

        # Step 4: Compute score-weighted linear combination of activation maps
        acts_np = acts.cpu()
        cam = torch.sum(score_weights.cpu() * acts_np, dim=0).numpy()
        cam = np.maximum(cam, 0.0)

        if cam.max() > 0:
            cam = cam / cam.max()

        heatmap_resized = cv2.resize(cam, (w_in, h_in))
        return heatmap_resized

    def save_visualization(
        self,
        input_tensor: torch.Tensor,
        original_rgb: np.ndarray,
        save_path: str | Path,
        target_class: int | None = None,
        alpha: float = 0.5,
    ) -> Path:
        """Generate and save gradient-free Score-CAM visualization overlay to disk."""
        heatmap = self.generate(input_tensor=input_tensor, target_class=target_class)
        overlay = overlay_heatmap(original_rgb, heatmap, alpha=alpha)

        out_path = Path(save_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out_path), bgr)
        return out_path
