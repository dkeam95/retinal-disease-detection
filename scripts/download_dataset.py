"""Dataset Download Utility for Retinal Disease Detection Framework.

Downloads the official DDR (Diabetic Retinopathy) Dataset from Hugging Face
(`addone5/DDR-dataset`) into the project's expected target directory (`data/raw`).

Usage:
    python scripts/download_dataset.py
    python scripts/download_dataset.py --output-dir data/raw
    python scripts/download_dataset.py --verify-only
"""

from __future__ import annotations

import argparse
import os
import sys
import zipfile
from pathlib import Path

# Ensure src/ is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

DEFAULT_REPO_ID = "addone5/DDR-dataset"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "raw"


def ensure_huggingface_hub() -> bool:
    """Checks if huggingface_hub is installed, attempting auto-install if missing."""
    try:
        import huggingface_hub  # noqa: F401

        return True
    except ImportError:
        print("[INIT] 'huggingface_hub' package not found. Attempting auto-installation...")
        import subprocess

        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "huggingface_hub"],
                check=True,
            )
            print("[SUCCESS] Successfully installed 'huggingface_hub'.")
            return True
        except subprocess.CalledProcessError as err:
            print(f"[ERROR] Automatic installation failed: {err}")
            print("Please run: pip install huggingface_hub")
            return False


def extract_archives(output_dir: Path) -> None:
    """Extracts any zip archives found in output_dir to their target folders."""
    zip_files = list(output_dir.glob("*.zip"))
    if not zip_files:
        return

    print(f"\n[POST-PROCESSING] Found {len(zip_files)} zip archive(s). Extracting...")
    for zip_path in zip_files:
        target_folder_name = zip_path.stem  # e.g., 'train', 'valid', 'test'
        extract_to = output_dir / target_folder_name
        extract_to.mkdir(parents=True, exist_ok=True)

        print(f" -> Extracting '{zip_path.name}' to '{extract_to.relative_to(PROJECT_ROOT)}'...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_to)
        print(f" [OK] Extracted '{zip_path.name}'.")


def verify_dataset_structure(data_dir: Path) -> bool:
    """Verifies that the target dataset directory matches expected project structure."""
    print("\n" + "=" * 65)
    print(" DATASET STRUCTURE INTEGRITY CHECK")
    print("=" * 65)
    print(f" Root Path: {data_dir.resolve()}")

    expected_splits = ["train", "valid", "test"]
    expected_annotations = ["train.txt", "valid.txt", "test.txt"]

    all_ok = True

    # 1. Check annotation text files
    print("\n [1/2] Annotation Text Files:")
    for ann in expected_annotations:
        ann_path = data_dir / ann
        if ann_path.exists():
            line_count = sum(1 for line in ann_path.read_text(encoding="utf-8").splitlines() if line.strip())
            print(f"   [OK] {ann:12s} : FOUND ({line_count} samples)")
        else:
            # Check recursive search fallback
            found = list(data_dir.rglob(ann))
            if found:
                print(f"   [OK] {ann:12s} : FOUND at {found[0].relative_to(data_dir)}")
            else:
                print(f"   [MISSING] {ann:12s} : NOT FOUND")
                all_ok = False

    # 2. Check image split directories
    print("\n [2/2] Image Split Directories:")
    for split in expected_splits:
        split_dir = data_dir / split
        if not split_dir.exists():
            found_dirs = list(data_dir.rglob(split))
            split_dir = found_dirs[0] if found_dirs else split_dir

        if split_dir.exists() and split_dir.is_dir():
            images = (
                list(split_dir.glob("*.jpg"))
                + list(split_dir.glob("*.jpeg"))
                + list(split_dir.glob("*.png"))
            )
            print(f"   [OK] {split + '/':12s} : FOUND ({len(images)} images)")
        else:
            print(f"   [MISSING] {split + '/':12s} : NOT FOUND")
            all_ok = False

    print("=" * 65)
    if all_ok:
        print(" SUCCESS: Dataset is complete and ready for training/evaluation!")
    else:
        print(" WARNING: Some dataset components are missing or incomplete.")
    print("=" * 65 + "\n")

    return all_ok


def download_ddr_dataset(
    repo_id: str = DEFAULT_REPO_ID,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    token: str | None = None,
    verify_only: bool = False,
) -> bool:
    """Downloads the DDR dataset from Hugging Face and verifies output integrity."""
    output_dir = Path(output_dir).resolve()

    if verify_only:
        return verify_dataset_structure(output_dir)

    if not ensure_huggingface_hub():
        return False

    from huggingface_hub import snapshot_download

    print("\n" + "=" * 65)
    print(" HUGGING FACE DATASET DOWNLOADER")
    print("=" * 65)
    print(f" Repository ID : {repo_id}")
    print(f" Output Directory: {output_dir}")
    print("=" * 65 + "\n")

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        print(f"[DOWNLOADING] Pulling dataset files from '{repo_id}'...")
        downloaded_path = snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=output_dir,
            token=token,
            ignore_patterns=["*.git*", "*.gitattributes"],
        )
        print(f"[SUCCESS] Download completed. Local path: {downloaded_path}")

        # Post-process any archives if present
        extract_archives(output_dir)

        # Verify final layout
        return verify_dataset_structure(output_dir)

    except Exception as e:
        print(f"\n[ERROR] Failed to download dataset: {e}")
        print("\nManual Alternative:")
        print(f"  1. Visit: https://huggingface.co/datasets/{repo_id}")
        print(f"  2. Extract files directly into: {output_dir}\n")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and verify DDR dataset from Hugging Face",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=DEFAULT_REPO_ID,
        help=f"Hugging Face dataset repository ID (default: {DEFAULT_REPO_ID})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Target local output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Optional Hugging Face authentication token",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify dataset structure without triggering download",
    )

    args = parser.parse_args()

    success = download_ddr_dataset(
        repo_id=args.repo_id,
        output_dir=args.output_dir,
        token=args.token,
        verify_only=args.verify_only,
    )

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
