#!/usr/bin/env python3
"""CLI script for downloading and setting up public datasets.

Provides download instructions, directory structure creation, and
optional TCIA REST API integration for the Digital Twin Tumor system.

Usage:
    python scripts/download_datasets.py --list
    python scripts/download_datasets.py --dataset proteas_brain_met
    python scripts/download_datasets.py --dataset rider_lung_ct --create-dirs
    python scripts/download_datasets.py --all
    python scripts/download_datasets.py --validate --dataset proteas_brain_met --data-dir data/proteas_brain_met
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

# Ensure the project source is importable
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root / "src"))

from digital_twin_tumor.data.datasets import (
    DATASETS,
    DatasetRegistry,
    dataset_download_instructions,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# TCIA REST API base URL
TCIA_API_BASE = "https://services.cancerimagingarchive.net/nbia-api/services/v1"


def _tcia_api_get(endpoint: str) -> list[dict] | None:
    """Make a GET request to the TCIA REST API.

    Returns parsed JSON or None on failure.
    """
    url = f"{TCIA_API_BASE}/{endpoint}"
    req = Request(url, headers={"Accept": "application/json"})
    try:
        with urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data
    except (URLError, TimeoutError, json.JSONDecodeError) as exc:
        logger.warning("TCIA API request failed (%s): %s", endpoint, exc)
        return None


def list_datasets() -> None:
    """Print a summary of all supported datasets."""
    print("\n" + "=" * 70)
    print("  Supported Public Datasets for Digital Twin Tumor")
    print("=" * 70)

    tier_map = {
        "proteas_brain_met": "A (Core)",
        "rider_lung_ct": "B (Calibration)",
        "qin_lungct_seg": "B (Benchmark)",
    }

    for key, info in DATASETS.items():
        tier = tier_map.get(key, "C")
        longitudinal = "Yes" if info["longitudinal"] else "No"
        print(f"\n  [{key}]")
        print(f"    Name:         {info['name']}")
        print(f"    Tier:         {tier}")
        print(f"    Source:       {info['source']}")
        print(f"    Modality:     {info['modality']}")
        print(f"    Patients:     {info['patients']}")
        print(f"    Longitudinal: {longitudinal}")
        print(f"    URL:          {info['url']}")

    print("\n" + "=" * 70)


def show_instructions(dataset_key: str | None) -> None:
    """Print download instructions for a specific dataset or all."""
    if dataset_key is None:
        print(dataset_download_instructions())
        return

    info = DATASETS.get(dataset_key)
    if info is None:
        logger.error("Unknown dataset: %s", dataset_key)
        logger.info("Available: %s", ", ".join(DATASETS.keys()))
        sys.exit(1)

    print(f"\n{'=' * 60}")
    print(f"  Download Instructions: {info['name']}")
    print(f"{'=' * 60}")
    print(f"\n  Source: {info['source']}")
    print(f"  URL:    {info['url']}")
    print(f"\n  Steps:")
    print(f"  {info['download_instructions']}")

    # Try TCIA API for TCIA datasets
    tcia_collection = info.get("tcia_collection")
    if tcia_collection:
        print(f"\n  Checking TCIA API for collection '{tcia_collection}'...")
        _check_tcia_collection(tcia_collection)

    print(f"\n  Citation:")
    print(f"  {info.get('citation', 'See URL for citation.')}")
    print()


def _check_tcia_collection(collection_name: str) -> None:
    """Query TCIA API for collection info and print results."""
    # Check if collection is available
    collections = _tcia_api_get("getCollectionValues")
    if collections is None:
        print("    [!] Could not reach TCIA API. Use manual download instead.")
        return

    collection_names = [
        c.get("Collection", "") for c in collections
    ]
    matched = [c for c in collection_names if collection_name.lower() in c.lower()]

    if matched:
        print(f"    [OK] Collection found on TCIA: {matched[0]}")

        # Try to get patient count
        patients = _tcia_api_get(
            f"getPatient?Collection={matched[0]}"
        )
        if patients:
            print(f"    [OK] Patients available: {len(patients)}")

        # Get series count
        series = _tcia_api_get(
            f"getSeries?Collection={matched[0]}"
        )
        if series:
            total_images = sum(
                int(s.get("ImageCount", 0)) for s in series
            )
            print(f"    [OK] Series: {len(series)}, Total images: {total_images}")

        print(f"\n    To download via NBIA Data Retriever:")
        print(f"    1. Go to: https://nbia.cancerimagingarchive.net/")
        print(f"    2. Search for collection: {matched[0]}")
        print(f"    3. Add to cart and download manifest (.tcia file)")
        print(f"    4. Open manifest with NBIA Data Retriever")
    else:
        print(f"    [!] Collection '{collection_name}' not found via API.")
        print(f"    Available collections with similar names:")
        for c in collection_names:
            if any(
                word.lower() in c.lower()
                for word in collection_name.split()
            ):
                print(f"      - {c}")


def create_directory_structure(dataset_key: str, base_dir: str) -> None:
    """Create the expected directory structure for a dataset."""
    info = DATASETS.get(dataset_key)
    if info is None:
        logger.error("Unknown dataset: %s", dataset_key)
        sys.exit(1)

    expected = info.get("expected_structure", {})
    root_dir = Path(base_dir) / expected.get("root", f"data/{dataset_key}/")

    print(f"\nCreating directory structure for {info['name']}...")
    print(f"  Root: {root_dir}")

    root_dir.mkdir(parents=True, exist_ok=True)
    print(f"  [OK] Created: {root_dir}")

    for subdir in expected.get("subdirs", []):
        # Replace glob patterns with example directories
        clean = subdir.replace("*", "example")
        full_path = root_dir / clean
        full_path.mkdir(parents=True, exist_ok=True)
        print(f"  [OK] Created: {full_path}")

    # Create a README in the data directory
    readme_path = root_dir / "README.txt"
    with open(readme_path, "w", encoding="utf-8") as fh:
        fh.write(f"Dataset: {info['name']}\n")
        fh.write(f"Source: {info['source']}\n")
        fh.write(f"URL: {info['url']}\n\n")
        fh.write("Download Instructions:\n")
        fh.write(info["download_instructions"])
        fh.write("\n")

    print(f"  [OK] Created README: {readme_path}")
    print(f"\n  Next: download the dataset files into {root_dir}")


def validate_dataset(dataset_key: str, data_dir: str) -> None:
    """Validate that a dataset directory has the expected structure."""
    result = DatasetRegistry.validate_directory(dataset_key, data_dir)

    print(f"\nValidation for '{dataset_key}' at {data_dir}:")
    print(f"  Status: {'VALID' if result['valid'] else 'INCOMPLETE'}")

    if result["found"]:
        print(f"\n  Found:")
        for item in result["found"]:
            print(f"    [OK] {item}")

    if result["missing"]:
        print(f"\n  Missing:")
        for item in result["missing"]:
            print(f"    [!!] {item}")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Download and set up public datasets for Digital Twin Tumor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/download_datasets.py --list\n"
            "  python scripts/download_datasets.py --dataset proteas_brain_met\n"
            "  python scripts/download_datasets.py --dataset rider_lung_ct --create-dirs\n"
            "  python scripts/download_datasets.py --validate "
            "--dataset proteas_brain_met --data-dir data/proteas_brain_met\n"
        ),
    )

    parser.add_argument(
        "--list", action="store_true",
        help="List all supported datasets with summaries",
    )
    parser.add_argument(
        "--dataset", type=str, default=None,
        help="Dataset key (e.g., proteas_brain_met, rider_lung_ct)",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Show instructions for all datasets",
    )
    parser.add_argument(
        "--create-dirs", action="store_true",
        help="Create the expected directory structure for --dataset",
    )
    parser.add_argument(
        "--validate", action="store_true",
        help="Validate a downloaded dataset directory",
    )
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Path to the dataset directory (for --validate)",
    )
    parser.add_argument(
        "--base-dir", type=str, default=".",
        help="Base directory for --create-dirs (default: current directory)",
    )

    args = parser.parse_args()

    if args.list:
        list_datasets()
        return

    if args.all:
        show_instructions(None)
        return

    if args.validate:
        if not args.dataset or not args.data_dir:
            parser.error("--validate requires --dataset and --data-dir")
        validate_dataset(args.dataset, args.data_dir)
        return

    if args.create_dirs:
        if not args.dataset:
            parser.error("--create-dirs requires --dataset")
        create_directory_structure(args.dataset, args.base_dir)
        return

    if args.dataset:
        show_instructions(args.dataset)
        return

    # No specific action requested -- show help
    parser.print_help()


if __name__ == "__main__":
    main()
