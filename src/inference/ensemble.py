"""Multi-Model Ensemble Predictor for Retinal Disease Classification."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from common.config.loader import ConfigLoader
from common.config.types import ProjectConfig
from inference.predictor import DEFAULT_CLASS_NAMES, PredictionResult, RetinalPredictor


class EnsemblePredictor:
    """Combines predictions from multiple fine-tuned models via weighted probability averaging."""

    def __init__(
        self,
        predictors: list[RetinalPredictor],
        weights: list[float] | None = None,
        class_names: list[str] | None = None,
    ) -> None:
        """Initialize Ensemble Predictor.

        Args:
            predictors: List of initialized RetinalPredictor instances.
            weights: Optional list of relative importance weights for each predictor.
            class_names: Target class names list.
        """
        if not predictors:
            raise ValueError(
                "EnsemblePredictor requires at least one RetinalPredictor instance."
            )

        self.predictors = predictors
        if weights is None:
            self.weights = [1.0 / len(predictors)] * len(predictors)
        else:
            if len(weights) != len(predictors):
                raise ValueError(
                    "Length of weights list must match length of predictors."
                )
            total_weight = sum(weights)
            self.weights = [w / total_weight for w in weights]

        self.class_names = (
            class_names if class_names is not None else DEFAULT_CLASS_NAMES
        )

    @classmethod
    def from_checkpoints(
        cls,
        checkpoint_config_pairs: list[tuple[str | Path, str | Path]],
        weights: list[float] | None = None,
        device: torch.device | str = "cpu",
    ) -> EnsemblePredictor:
        """Instantiate ensemble directly from pairs of (checkpoint_path, config_path).

        Args:
            checkpoint_config_pairs: List of tuples containing (checkpoint_file, yaml_config_file).
            weights: Optional relative ensemble weights.
            device: Compute device.

        Returns:
            EnsemblePredictor instance.
        """
        predictors = []
        for ckpt, cfg_path in checkpoint_config_pairs:
            config = ConfigLoader.load(cfg_path)
            predictor = RetinalPredictor.from_checkpoint(
                checkpoint_path=ckpt,
                config=config,
                device=device,
            )
            predictors.append(predictor)

        return cls(predictors=predictors, weights=weights)

    def predict(
        self, image_input: str | Path | np.ndarray | Image.Image
    ) -> PredictionResult:
        """Run ensemble prediction on a single image.

        Args:
            image_input: Image path, numpy RGB array, or PIL image.

        Returns:
            PredictionResult with averaged probability distribution.
        """
        all_probs = []
        for predictor, weight in zip(self.predictors, self.weights):
            res = predictor.predict(image_input)
            all_probs.append(np.array(res.probabilities) * weight)

        ensemble_probs = np.sum(all_probs, axis=0)
        grade_id = int(np.argmax(ensemble_probs))
        confidence = float(ensemble_probs[grade_id])
        grade_name = (
            self.class_names[grade_id]
            if grade_id < len(self.class_names)
            else f"Grade {grade_id}"
        )

        return PredictionResult(
            grade_id=grade_id,
            grade_name=grade_name,
            confidence=confidence,
            probabilities=[float(p) for p in ensemble_probs],
        )

    def predict_batch(
        self, image_inputs: list[str | Path | np.ndarray | Image.Image]
    ) -> list[PredictionResult]:
        """Run ensemble prediction over a batch of images.

        Args:
            image_inputs: List of image paths, arrays, or PIL images.

        Returns:
            List of PredictionResult objects.
        """
        return [self.predict(img) for img in image_inputs]
