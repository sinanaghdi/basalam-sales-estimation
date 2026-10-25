"""Download the pinned BaSalam archive, verify it, and convert to Parquet.

Usage: python scripts/prepare_data.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from basalam import data as data_mod  # noqa: E402


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    data_dir = repo / "data"
    archive = data_mod.download_archive(data_dir)
    csvs = data_mod.extract_archive(archive, data_dir)
    products_csv = data_dir / "BaSalam.products.csv"
    if not products_csv.exists():
        found = [c.name for c in csvs]
        raise FileNotFoundError(f"products CSV not found after extraction; found: {found}")
    data_mod.products_to_parquet(products_csv, data_dir / "products.parquet")
    data_mod.write_manifest(repo / "manifests" / "data_manifest.json")
    print("OK: data prepared and pinned in manifests/data_manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
