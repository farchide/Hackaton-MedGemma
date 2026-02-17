#!/usr/bin/env python3
"""Generate synthetic demo data for the Digital Twin Tumor system.

Usage:
    PYTHONPATH=src python3 scripts/generate_demo_data.py [--db-path PATH] [--seed N]

Creates realistic longitudinal oncology data for five patient scenarios
and stores everything in an SQLite database ready for the demo UI.
"""

from __future__ import annotations

import argparse
import sys
import time


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate synthetic demo data for Digital Twin Tumor.",
    )
    parser.add_argument(
        "--db-path",
        default=".cache/demo.db",
        help="Path to SQLite database (default: .cache/demo.db)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible generation (default: 42)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )
    args = parser.parse_args()

    print(f"Digital Twin Tumor -- Synthetic Data Generator")
    print(f"Database: {args.db_path}")
    print(f"Seed: {args.seed}")

    start = time.time()

    from digital_twin_tumor.data.synthetic import generate_all_demo_data

    results = generate_all_demo_data(
        db_path=args.db_path,
        seed=args.seed,
        verbose=not args.quiet,
    )

    elapsed = time.time() - start
    print(f"\nDone in {elapsed:.1f}s. Database written to: {args.db_path}")

    # Quick verification: re-open database and count rows
    from digital_twin_tumor.storage.sqlite_backend import SQLiteBackend
    db = SQLiteBackend(args.db_path)
    counts = {}
    for table in [
        "patients", "timepoints", "lesions", "measurements",
        "therapy_events", "recist_responses", "growth_models",
        "simulation_results",
    ]:
        rows = db.query(f"SELECT COUNT(*) as cnt FROM {table}")
        counts[table] = rows[0]["cnt"]
    db.close()

    print(f"\nDatabase row counts:")
    for table, count in counts.items():
        print(f"  {table:<25s}: {count:>6d}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
