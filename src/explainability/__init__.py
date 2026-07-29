"""Explainability package (Grad-CAM / Class Activation Mapping) for retinal fundus images."""

from .gradcam import GradCAM, overlay_heatmap

__all__ = ["GradCAM", "overlay_heatmap"]
