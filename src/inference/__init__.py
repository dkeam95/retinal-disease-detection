"""Inference package for single and batch fundus image predictions."""

from .detection_predictor import DetectionResult, LesionDetectionPredictor
from .masked_predictor import MaskedDetectionResult, MaskedLesionPredictor
from .predictor import PredictionResult, RetinalPredictor

__all__ = [
    "RetinalPredictor",
    "PredictionResult",
    "LesionDetectionPredictor",
    "DetectionResult",
    "MaskedLesionPredictor",
    "MaskedDetectionResult",
]
