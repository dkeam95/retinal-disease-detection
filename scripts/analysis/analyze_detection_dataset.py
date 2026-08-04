"""
XML Dataset Analyzer for Faster R-CNN Lesion Detection.

Analyzes:
- number of bounding boxes per XML
- image resolutions
- lesion class distribution
- top heaviest XML annotations

Usage:
    python scripts/analysis/analyze_detection_dataset.py
"""

from __future__ import annotations

import statistics
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import pandas as pd

XML_DIR = Path("data/raw/lesion_detection/train")


def analyze_xml(xml_path: Path) -> dict:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    filename = root.findtext("filename")

    width = int(root.find("size/width").text)
    height = int(root.find("size/height").text)

    boxes = root.findall("object")

    class_counter = Counter()

    areas = []

    for obj in boxes:
        cls = obj.findtext("name")

        box = obj.find("bndbox")

        xmin = int(box.findtext("xmin"))
        ymin = int(box.findtext("ymin"))
        xmax = int(box.findtext("xmax"))
        ymax = int(box.findtext("ymax"))

        class_counter[cls] += 1

        areas.append((xmax - xmin) * (ymax - ymin))

    return {
        "xml": xml_path.name,
        "image": filename,
        "width": width,
        "height": height,
        "boxes": len(boxes),
        "ex": class_counter["ex"],
        "he": class_counter["he"],
        "ma": class_counter["ma"],
        "se": class_counter["se"],
        "mean_box_area": statistics.mean(areas) if areas else 0,
        "median_box_area": statistics.median(areas) if areas else 0,
        "max_box_area": max(areas) if areas else 0,
    }


def percentile(values: list[int], p: float) -> float:
    values = sorted(values)

    k = (len(values) - 1) * p

    f = int(k)
    c = min(f + 1, len(values) - 1)

    if f == c:
        return values[f]

    return values[f] * (c - k) + values[c] * (k - f)


def main():

    xml_files = sorted(XML_DIR.glob("*.xml"))

    rows = [analyze_xml(x) for x in xml_files]

    df = pd.DataFrame(rows)

    output_path = Path("reports/metrics/detection_dataset_statistics.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    boxes = df["boxes"].tolist()

    print("=" * 80)
    print("DATASET SUMMARY")
    print("=" * 80)

    print(f"XML files      : {len(df)}")
    print(f"Min boxes      : {min(boxes)}")
    print(f"Median boxes   : {statistics.median(boxes):.1f}")
    print(f"Mean boxes     : {statistics.mean(boxes):.1f}")
    print(f"95 percentile  : {percentile(boxes,0.95):.1f}")
    print(f"99 percentile  : {percentile(boxes,0.99):.1f}")
    print(f"Max boxes      : {max(boxes)}")

    print()

    print("=" * 80)
    print("TOP 20 HEAVIEST XML FILES")
    print("=" * 80)

    print(
        df.sort_values("boxes", ascending=False)[
            [
                "xml",
                "boxes",
                "width",
                "height",
                "ex",
                "he",
                "ma",
                "se",
            ]
        ].head(20)
    )

    print()

    print("=" * 80)
    print("CLASS TOTALS")
    print("=" * 80)

    print(f"EX : {df['ex'].sum()}")
    print(f"HE : {df['he'].sum()}")
    print(f"MA : {df['ma'].sum()}")
    print(f"SE : {df['se'].sum()}")

    print()

    print("CSV saved as reports/metrics/detection_dataset_statistics.csv")


if __name__ == "__main__":
    main()
