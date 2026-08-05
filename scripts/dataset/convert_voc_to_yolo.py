"""PASCAL VOC XML to YOLO Dataset Converter with High-Resolution Tiling for Retinal Lesion Detection.

Converts PASCAL VOC XML bounding box annotations into YOLO format (class_id cx cy w h).
Supports cropping high-resolution fundus images (3000x3000) into overlapping 1024x1024 tiles
to preserve micro-lesions (Microaneurysms and Hard Exudates).
"""

from __future__ import annotations

import argparse
import logging
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

CLASS_MAP = {
    "ex": 0,  # Hard Exudates
    "he": 1,  # Hemorrhages
    "ma": 2,  # Microaneurysms
    "se": 3,  # Soft Exudates
}

CLASS_NAMES = ["Hard Exudate (EX)", "Hemorrhage (HE)", "Microaneurysm (MA)", "Soft Exudate (SE)"]


def parse_voc_xml(xml_path: Path) -> dict:
    """Parse PASCAL VOC XML annotation file."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size_elem = root.find("size")
    width = int(size_elem.find("width").text) if size_elem is not None else 0
    height = int(size_elem.find("height").text) if size_elem is not None else 0

    boxes = []
    for obj in root.findall("object"):
        cls_name = obj.findtext("name", "").strip().lower()
        if cls_name not in CLASS_MAP:
            continue
        cls_id = CLASS_MAP[cls_name]

        bnd = obj.find("bndbox")
        if bnd is None:
            continue

        xmin = float(bnd.findtext("xmin"))
        ymin = float(bnd.findtext("ymin"))
        xmax = float(bnd.findtext("xmax"))
        ymax = float(bnd.findtext("ymax"))

        if xmax > xmin and ymax > ymin:
            boxes.append([cls_id, xmin, ymin, xmax, ymax])

    return {"width": width, "height": height, "boxes": boxes}


def process_split_tiled(
    xml_dir: Path,
    image_dir: Path,
    output_img_dir: Path,
    output_lbl_dir: Path,
    tile_size: int = 1024,
    overlap: int = 256,
    min_box_size: float = 2.0,
) -> int:
    """Process split by slicing images into overlapping tiles and adjusting box coordinates."""
    output_img_dir.mkdir(parents=True, exist_ok=True)
    output_lbl_dir.mkdir(parents=True, exist_ok=True)

    xml_files = sorted(list(xml_dir.glob("*.xml")))
    stride = tile_size - overlap
    total_saved_tiles = 0

    for xml_path in xml_files:
        stem = xml_path.stem
        img_path = image_dir / f"{stem}.jpg"
        if not img_path.exists():
            img_path = image_dir / f"{stem}.png"
        if not img_path.exists():
            continue

        bgr = cv2.imread(str(img_path))
        if bgr is None:
            continue

        h, w = bgr.shape[:2]
        anno = parse_voc_xml(xml_path)
        gt_boxes = anno["boxes"]

        tile_idx = 0
        y_start = 0
        while y_start < h:
            y_end = min(y_start + tile_size, h)
            x_start = 0
            while x_start < w:
                x_end = min(x_start + tile_size, w)

                tile_boxes = []
                for cls_id, x1, y1, x2, y2 in gt_boxes:
                    # Clip box to tile boundaries
                    tx1 = max(x1, float(x_start))
                    ty1 = max(y1, float(y_start))
                    tx2 = min(x2, float(x_end))
                    ty2 = min(y2, float(y_end))

                    tile_w = tx2 - tx1
                    tile_h = ty2 - ty1

                    # Keep box if it retains significant area in tile
                    orig_area = (x2 - x1) * (y2 - y1)
                    cropped_area = tile_w * tile_h

                    if tile_w >= min_box_size and tile_h >= min_box_size and cropped_area >= 0.3 * orig_area:
                        # Convert to tile-local coordinates
                        lx1 = tx1 - x_start
                        ly1 = ty1 - y_start
                        lx2 = tx2 - x_start
                        ly2 = ty2 - y_start

                        curr_tile_w = x_end - x_start
                        curr_tile_h = y_end - y_start

                        # YOLO normalized coordinates: cx, cy, w, h
                        cx = ((lx1 + lx2) / 2.0) / curr_tile_w
                        cy = ((ly1 + ly2) / 2.0) / curr_tile_h
                        nw = tile_w / curr_tile_w
                        nh = tile_h / curr_tile_h

                        tile_boxes.append(f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

                # Save tile image and labels (only if tile contains boxes or for negative sampling)
                if tile_boxes:
                    tile_img = bgr[y_start:y_end, x_start:x_end]
                    tile_name = f"{stem}_tile_{tile_idx}"
                    tile_img_file = output_img_dir / f"{tile_name}.jpg"
                    tile_lbl_file = output_lbl_dir / f"{tile_name}.txt"

                    cv2.imwrite(str(tile_img_file), tile_img)
                    with open(tile_lbl_file, "w", encoding="utf-8") as f:
                        f.write("\n".join(tile_boxes) + "\n")

                    total_saved_tiles += 1
                    tile_idx += 1

                if x_end == w:
                    break
                x_start += stride

            if y_end == h:
                break
            y_start += stride

    return total_saved_tiles


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert VOC XML annotations to YOLO format with tiling")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/raw/lesion_detection"),
        help="Path to lesion_detection input directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/yolo_lesions_1024"),
        help="Output directory for processed YOLO dataset",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=1024,
        help="Tile crop resolution (default: 1024)",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=256,
        help="Overlap stride between tiles (default: 256)",
    )
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir

    logger.info(f"Converting VOC XMLs from {input_dir} to YOLO dataset at {output_dir}")

    splits = ["train", "valid", "test"]
    for split in splits:
        xml_dir = input_dir / split
        if not xml_dir.exists():
            logger.warning(f"Split directory missing: {xml_dir}, skipping...")
            continue

        image_dir = input_dir.parent / split
        out_img_dir = output_dir / "images" / split
        out_lbl_dir = output_dir / "labels" / split

        count = process_split_tiled(
            xml_dir=xml_dir,
            image_dir=image_dir,
            output_img_dir=out_img_dir,
            output_lbl_dir=out_lbl_dir,
            tile_size=args.tile_size,
            overlap=args.overlap,
        )
        logger.info(f"Split [{split}]: Generated {count} tiles with bounding boxes.")

    # Create YOLO dataset.yaml
    dataset_yaml = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/valid",
        "test": "images/test",
        "names": {i: name for i, name in enumerate(CLASS_NAMES)},
    }

    yaml_path = output_dir / "dataset.yaml"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(dataset_yaml, f, sort_keys=False)

    logger.info(f"YOLO dataset config saved to: {yaml_path}")


if __name__ == "__main__":
    main()
