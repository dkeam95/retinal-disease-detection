"""Integrated Gradients (Axiomatic Pixel-Level Feature Attribution) implementation.

Paper: Axiomatic Attribution for Deep Networks
       Sundararajan et al., 2017 (ICML) (https://arxiv.org/abs/1703.01365)

Integrated Gradients computes the path integral of gradients of the target class logit with respect
to the input image tensor along a linear interpolation path from a baseline image (e.g. black image)
to the target input image. It is an attribution method, not a lesion segmentation model.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

from explainability.gradcam import overlay_heatmap


class IntegratedGradients:
    """Integrated Gradients pixel-level feature attribution engine."""

    def __init__(
        self,
        model: nn.Module,
        n_steps: int = 30,
        normalization_mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        normalization_std: tuple[float, float, float] = (0.229, 0.224, 0.225),
    ) -> None:
        """Initialize Integrated Gradients.

        Args:
            model: PyTorch neural network model.
            n_steps: Number of linear interpolation steps between baseline and input (default: 30).
            normalization_mean: Mean used to normalize input RGB images.
            normalization_std: Standard deviation used to normalize input RGB images.
        """
        self.model = model
        self.n_steps = n_steps
        self.normalization_mean = normalization_mean
        self.normalization_std = normalization_std

    def _default_baseline(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Return a black RGB image expressed in normalized input space."""
        if input_tensor.ndim != 4 or input_tensor.shape[1] != 3:
            raise ValueError("Integrated Gradients expects input shape (N, 3, H, W).")

        mean = torch.as_tensor(
            self.normalization_mean, device=input_tensor.device, dtype=input_tensor.dtype
        ).view(1, 3, 1, 1)
        std = torch.as_tensor(
            self.normalization_std, device=input_tensor.device, dtype=input_tensor.dtype
        ).view(1, 3, 1, 1)
        return (-mean / std).expand_as(input_tensor).clone()

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: int | None = None,
        baseline_tensor: torch.Tensor | None = None,
    ) -> np.ndarray:
        """Generate pixel-level Integrated Gradients attribution map for an input image tensor.

        Args:
            input_tensor: Preprocessed image tensor of shape (1, C, H, W).
            target_class: Target class index. If None, uses top predicted class.
            baseline_tensor: Optional baseline tensor of shape (1, C, H, W). If None,
                uses black RGB represented in ImageNet-normalized tensor space.

        Returns:
            Normalised 2D numpy array attribution map (H, W) in range [0.0, 1.0].
        """
        self.model.eval()
        device = input_tensor.device

        # Step 1: Establish baseline. Zero is the ImageNet mean colour, not black.
        if baseline_tensor is None:
            baseline_tensor = self._default_baseline(input_tensor)
        else:
            baseline_tensor = baseline_tensor.to(device)

        # Determine target class if None
        if target_class is None:
            with torch.no_grad():
                outputs = self.model(input_tensor)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs
                target_class = int(torch.argmax(logits, dim=1).item())

        # Step 2: Generate linearly interpolated path inputs from baseline to input
        alphas = torch.linspace(0.0, 1.0, steps=self.n_steps + 1, device=device)
        # Difference vector (x - x')
        diff = input_tensor - baseline_tensor  # (1, C, H, W)

        accumulated_grads = torch.zeros_like(input_tensor)

        # Step 3: Compute gradients for each interpolated image step
        for step, alpha in enumerate(alphas):
            interpolated_input = baseline_tensor + alpha * diff
            interpolated_input = (
                interpolated_input.clone().detach().requires_grad_(True)
            )

            self.model.zero_grad()
            outputs = self.model(interpolated_input)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs
            score = logits[0, target_class]
            score.backward()

            if interpolated_input.grad is not None:
                endpoint_weight = 0.5 if step in (0, self.n_steps) else 1.0
                accumulated_grads += endpoint_weight * interpolated_input.grad.detach()

        # Step 4: Average gradients over steps and multiply by difference (x - x')
        avg_grads = accumulated_grads / self.n_steps
        integrated_grads = diff * avg_grads  # (1, C, H, W)

        # Aggregate across RGB channels using magnitude (L2 norm across channel dim)
        ig_map = (
            torch.norm(integrated_grads.squeeze(0), dim=0).detach().cpu().numpy()
        )  # (H, W)

        # Step 5: Normalise to [0, 1]
        ig_min, ig_max = ig_map.min(), ig_map.max()
        if ig_max > ig_min:
            ig_norm = (ig_map - ig_min) / (ig_max - ig_min)
        else:
            ig_norm = np.zeros_like(ig_map)

        # Smooth slightly using a small Gaussian blur to enhance visualization continuity
        smoothed = cv2.GaussianBlur(ig_norm, (3, 3), 0)
        if smoothed.max() > 0:
            smoothed = smoothed / smoothed.max()

        return smoothed

    def save_visualization(
        self,
        input_tensor: torch.Tensor,
        original_rgb: np.ndarray,
        save_path: str | Path,
        target_class: int | None = None,
        baseline_tensor: torch.Tensor | None = None,
        alpha: float = 0.6,
    ) -> Path:
        """Generate and save pixel-level Integrated Gradients visualization overlay to disk."""
        heatmap = self.generate(
            input_tensor=input_tensor,
            target_class=target_class,
            baseline_tensor=baseline_tensor,
        )
        overlay = overlay_heatmap(original_rgb, heatmap, alpha=alpha)

        out_path = Path(save_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out_path), bgr)
        return out_path
