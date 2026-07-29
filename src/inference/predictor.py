"""High-level Retinal Disease Predictor for single image and batch inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image

from common.config.types import ProjectConfig
from preprocessing.pipeline import build_validation_pipeline
from training.model_factory import ModelFactory

DEFAULT_CLASS_NAMES = [
    "No DR",
    "Mild NPDR",
    "Moderate NPDR",
    "Severe NPDR",
    "Proliferative DR",
]


@dataclass(frozen=True, slots=True)
class PredictionResult:
    """Dataclass encapsulating prediction outputs for a fundus image.

    Attributes:
        grade_id: Predicted DR grade integer (0-4).
        grade_name: Clinical DR stage description name.
        confidence: Probability score for the top predicted grade.
        probabilities: Full probability distribution array across all 5 classes.
    """

    grade_id: int
    grade_name: str
    confidence: float
    probabilities: list[float]

    def to_dict(self) -> dict[str, Any]:
        """Convert prediction result to dictionary representation."""
        return {
            "grade_id": self.grade_id,
            "grade_name": self.grade_name,
            "confidence": self.confidence,
            "probabilities": self.probabilities,
        }


class RetinalPredictor:
    """Predictor engine for evaluating single images or image arrays."""

    def __init__(
        self,
        model: torch.nn.Module,
        config: ProjectConfig,
        device: torch.device | str = "cpu",
        class_names: list[str] | None = None,
    ) -> None:
        """Initialize Predictor.

        Args:
            model: Trained PyTorch neural network model.
            config: Project configuration containing preprocessing parameters.
            device: Computing device.
            class_names: List of class descriptions.
        """
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.model.eval()
        self.config = config
        self.class_names = class_names if class_names is not None else DEFAULT_CLASS_NAMES
        self.transform = build_validation_pipeline(config.preprocessing)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        config: ProjectConfig,
        device: torch.device | str = "cpu",
    ) -> RetinalPredictor:
        """Instantiate Predictor directly from a saved checkpoint file.

        Args:
            checkpoint_path: Path to PyTorch model checkpoint (.pt).
            config: Project configuration object.
            device: Computing device.

        Returns:
            RetinalPredictor instance.
        """
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {path}")

        dev = torch.device(device)
        model = ModelFactory.build(
            architecture=config.model.architecture,
            pretrained=False,
            num_classes=config.model.num_classes,
        )

        checkpoint_data = torch.load(path, map_location=dev, weights_only=False)
        state_dict = (
            checkpoint_data["model_state"]
            if isinstance(checkpoint_data, dict) and "model_state" in checkpoint_data
            else checkpoint_data
        )
        model.load_state_dict(state_dict)

        return cls(model=model, config=config, device=dev)

    def _prepare_image_tensor(self, image_input: str | Path | np.ndarray | Image.Image) -> torch.Tensor:
        """Load and preprocess image input into a batch tensor."""
        if isinstance(image_input, (str, Path)):
            path = Path(image_input)
            if not path.exists():
                raise FileNotFoundError(f"Input image not found: {path}")
            bgr = cv2.imread(str(path))
            if bgr is None:
                raise ValueError(f"Failed to read image at path: {path}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        elif isinstance(image_input, Image.Image):
            rgb = np.array(image_input.convert("RGB"))
        elif isinstance(image_input, np.ndarray):
            rgb = image_input if image_input.ndim == 3 and image_input.shape[2] == 3 else cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        augmented = self.transform(image=rgb)
        tensor = augmented["image"]  # Tensor (C, H, W)
        return tensor.unsqueeze(0).to(self.device)  # Tensor (1, C, H, W)

    @torch.no_grad()
    def predict(self, image_input: str | Path | np.ndarray | Image.Image) -> PredictionResult:
        """Run disease severity prediction on a single fundus image.

        Args:
            image_input: Image path, numpy RGB array, or PIL image.

        Returns:
            PredictionResult dataclass instance.
        """
        tensor = self._prepare_image_tensor(image_input)
        outputs = self.model(tensor)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs

        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        grade_id = int(np.argmax(probs))
        confidence = float(probs[grade_id])
        grade_name = self.class_names[grade_id] if grade_id < len(self.class_names) else f"Grade {grade_id}"

        return PredictionResult(
            grade_id=grade_id,
            grade_name=grade_name,
            confidence=confidence,
            probabilities=[float(p) for p in probs],
        )
