"""Inference package for single and batch fundus image predictions."""

from .predictor import PredictionResult, RetinalPredictor

__all__ = ["RetinalPredictor", "PredictionResult"]
