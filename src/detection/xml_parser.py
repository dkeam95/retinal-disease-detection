"""PASCAL VOC XML Annotation Parser for Diabetic Retinopathy Lesions.

Parses XML bounding box annotations for 4 lesion classes:
  - 1: 'ex' (Hard Exudates)
  - 2: 'he' (Hemorrhages)
  - 3: 'ma' (Microaneurysms)
  - 4: 'se' (Soft Exudates)
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

# Mapping from DDR XML class names to integer class IDs
LESION_CLASS_MAP: dict[str, int] = {
    "ex": 1,
    "he": 2,
    "ma": 3,
    "se": 4,
}

LESION_ID_TO_NAME: dict[int, str] = {
    0: "__background__",
    1: "Hard Exudate (EX)",
    2: "Hemorrhage (HE)",
    3: "Microaneurysm (MA)",
    4: "Soft Exudate (SE)",
}


@dataclass(frozen=True, slots=True)
class ParsedAnnotation:
    """Dataclass holding parsed XML bounding box metadata for a single image.

    Attributes:
        filename: Corresponding image filename (e.g. '007-1774-100.jpg').
        width: Image pixel width.
        height: Image pixel height.
        boxes: List of bounding boxes [xmin, ymin, xmax, ymax].
        labels: List of integer class IDs (1..4).
    """

    filename: str
    width: int
    height: int
    boxes: list[list[float]]
    labels: list[int]


def parse_voc_xml(xml_path: str | Path) -> ParsedAnnotation:
    """Parse a PASCAL VOC XML file into structured bounding boxes and labels.

    Args:
        xml_path: Path to the XML file.

    Returns:
        ParsedAnnotation containing filename, image dimensions, boxes, and labels.

    Raises:
        FileNotFoundError: If the XML file does not exist.
        ValueError: If the XML format is invalid.
    """
    path = Path(xml_path)
    if not path.exists():
        raise FileNotFoundError(f"XML annotation file not found: {path}")

    tree = ET.parse(path)
    root = tree.getroot()

    filename_elem = root.find("filename")
    filename = (
        filename_elem.text.strip()
        if filename_elem is not None and filename_elem.text
        else path.stem + ".jpg"
    )

    size_elem = root.find("size")
    width = int(size_elem.find("width").text) if size_elem is not None else 0
    height = int(size_elem.find("height").text) if size_elem is not None else 0

    boxes: list[list[float]] = []
    labels: list[int] = []

    for obj in root.findall("object"):
        name_elem = obj.find("name")
        if name_elem is None or not name_elem.text:
            continue

        raw_name = name_elem.text.strip().lower()
        if raw_name not in LESION_CLASS_MAP:
            continue

        class_id = LESION_CLASS_MAP[raw_name]
        bndbox = obj.find("bndbox")
        if bndbox is None:
            continue

        xmin = float(bndbox.find("xmin").text)
        ymin = float(bndbox.find("ymin").text)
        xmax = float(bndbox.find("xmax").text)
        ymax = float(bndbox.find("ymax").text)

        # Validate box coordinates: ensure min < max
        if xmax <= xmin or ymax <= ymin:
            continue

        # Clip box coordinates if image dimensions are present
        if width > 0 and height > 0:
            xmin = max(0.0, min(xmin, float(width - 1)))
            ymin = max(0.0, min(ymin, float(height - 1)))
            xmax = max(xmin + 1.0, min(xmax, float(width)))
            ymax = max(ymin + 1.0, min(ymax, float(height)))

        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(class_id)

    return ParsedAnnotation(
        filename=filename,
        width=width,
        height=height,
        boxes=boxes,
        labels=labels,
    )
