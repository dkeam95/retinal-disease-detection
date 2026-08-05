"""PyTorch Dataset implementation for Diabetic Retinopathy Lesion Detection."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from detection.xml_parser import parse_voc_xml


class LesionDetectionDataset(Dataset[tuple[torch.Tensor, dict[str, torch.Tensor]]]):
    """PyTorch Dataset loading fundus images and XML lesion bounding boxes for Object Detection."""

    def __init__(
        self,
        xml_dir: str | Path,
        image_dir: str | Path,
        transform: Callable[[np.ndarray, list[list[float]], list[int]], tuple[torch.Tensor, list[list[float]], list[int]]] | None = None,
        target_size: tuple[int, int] | None = None,
        enable_clahe: bool = True,
    ) -> None:
        """Initialize LesionDetectionDataset.

        Args:
            xml_dir: Directory containing PASCAL VOC XML annotations.
            image_dir: Directory containing corresponding JPG images.
            transform: Optional custom augmentation function.
            target_size: Deprecated compatibility argument. Resizing is performed by
                Faster R-CNN so it can preserve image and box geometry.
            enable_clahe: Whether to apply CLAHE contrast enhancement in LAB color space.
        """
        self.xml_dir = Path(xml_dir)
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.target_size = target_size
        self.enable_clahe = enable_clahe

        if not self.xml_dir.exists():
            raise FileNotFoundError(f"XML directory not found: {self.xml_dir}")

        # Gather all XML annotation files
        self.xml_files = sorted(list(self.xml_dir.glob("*.xml")))
        if not self.xml_files:
            raise ValueError(f"No XML files found in directory: {self.xml_dir}")

    def __len__(self) -> int:
        return len(self.xml_files)

    def _resolve_image_path(self, xml_filename: str, parsed_filename: str) -> Path:
        """Locate image file on disk in image_dir or fallback directories."""
        stem = Path(xml_filename).stem
        candidates = [
            self.image_dir / parsed_filename,
            self.image_dir / f"{stem}.jpg",
            self.image_dir / f"{stem}.png",
            self.xml_dir.parent.parent / "lesion_segmentation" / self.xml_dir.name / "image" / f"{stem}.jpg",
            self.image_dir.parent / "lesion_segmentation" / self.image_dir.name / "image" / f"{stem}.jpg",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"Image for {xml_filename} not found in candidates: {candidates}")

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        xml_path = self.xml_files[index]
        annotation = parse_voc_xml(xml_path)

        image_path = self._resolve_image_path(xml_path.name, annotation.filename)
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"Failed to read image at: {image_path}")

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        if self.enable_clahe:
            lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
            l_chan, a_chan, b_chan = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l_enhanced = clahe.apply(l_chan)
            lab_enhanced = cv2.merge((l_enhanced, a_chan, b_chan))
            rgb = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)

        boxes: list[list[float]] = []
        labels: list[int] = []

        for box, label in zip(annotation.boxes, annotation.labels, strict=False):
            xmin, ymin, xmax, ymax = map(float, box)

            if xmax > xmin + 1.0 and ymax > ymin + 1.0:
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(label)

        # GeneralizedRCNNTransform owns ImageNet normalization and aspect-ratio
        # preserving resize. Supplying only [0, 1] RGB prevents double normalization.
        img_tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0

        if boxes:
            boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32)
            labels_tensor = torch.as_tensor(labels, dtype=torch.int64)
            area_tensor = (boxes_tensor[:, 2] - boxes_tensor[:, 0]) * (
                boxes_tensor[:, 3] - boxes_tensor[:, 1]
            )
        else:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)
            area_tensor = torch.zeros((0,), dtype=torch.float32)

        target: dict[str, torch.Tensor] = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": torch.tensor([index], dtype=torch.int64),
            "area": area_tensor,
            "iscrowd": torch.zeros((len(labels_tensor),), dtype=torch.uint8),
        }

        return img_tensor, target


def detection_collate_fn(
    batch: list[tuple[torch.Tensor, dict[str, torch.Tensor]]],
) -> tuple[list[torch.Tensor], list[dict[str, torch.Tensor]]]:
    """Custom collate function for PyTorch DataLoader handling variable-length detection targets."""
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets
