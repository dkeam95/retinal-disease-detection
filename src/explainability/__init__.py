"""Explainability package: Grad-CAM, Grad-CAM++, Layer-CAM, Score-CAM, Integrated Gradients, CLAHE Spotlight, Lesion Segmenter."""

from .engine import ExplainabilityEngine
from .gradcam import GradCAM, overlay_heatmap
from .gradcam_plus_plus import GradCAMPlusPlus
from .integrated_gradients import IntegratedGradients
from .layer_cam import LayerCAM
from .lesion_segmenter import LesionSegmenter
from .score_cam import ScoreCAM
from .spotlight import SpotlightVisualizer, clahe_enhance, create_4panel, draw_spotlight

__all__ = [
    "ExplainabilityEngine",
    "GradCAM",
    "GradCAMPlusPlus",
    "LayerCAM",
    "ScoreCAM",
    "IntegratedGradients",
    "LesionSegmenter",
    "overlay_heatmap",
    "SpotlightVisualizer",
    "clahe_enhance",
    "create_4panel",
    "draw_spotlight",
]
