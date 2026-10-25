"""Data acquisition, verification, and auditing for the BaSalam dataset.

The raw data is NEVER committed to the repository. It is downloaded from
Hugging Face, verified against a pinned SHA-256 checksum, and converted to
typed Parquet once. All statistics in this module reproduce the audit
documented in the project decision log (2026-10-24).
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

# Pinned artifact (verified 2026-10-24 against the actual archive)
ARCHIVE_URL = (
    "https://huggingface.co/datasets/RadeAI/BaSalam_comments_products/"
    "resolve/main/BaSalam_dataset.rar"
)
ARCHIVE_SHA256 = "12e79a5bfe8322d9a084e327ecb79bc64ea5d232ce157daf0dd06cbad5ab0555"
ARCHIVE_BYTES = 434_741_588

PRODUCTS_CSV = "BaSalam.products.csv"
REVIEWS_CSV = "BaSalam.reviews.csv"

# Audited properties of the pinned artifact (see manifests/data_manifest.json)
AUDITED = {
    "products_rows": 2_411_358,
    "reviews_rows": 3_393_574,
    "distinct_vendors": 88_822,
    "missing_vendor_id": 3,
    "duplicate_product_id": 0,
    "zero_share": 0.9681,
    "total_weekly_sales": 105_540,
    "max_weekly_sales": 56,
    "nonzero_rows": 76_942,
}

# Full column schema of the pinned products CSV (audited 2026-10-24)
PRODUCTS_COLUMNS = [
    "_id", "_score", "sales_count_week", "name", "price", "status_id",
    "status_title", "stock", "photo_MEDIUM", "photo_SMALL",
    "rating_average", "rating_count", "rating_signals", "primaryPrice",
    "preparationDays", "weight", "categoryId", "has_delivery",
    "has_variation", "new_categoryId", "navigation_id", "vendor_name",
    "vendor_identifier", "vendor_statusId", "vendor_freeShippingToIran",
    "vendor_freeShippingToSameCity", "vendor_cityId", "vendor_provinceId",
    "vendor_has_delivery", "vendor_score", "vendor_id",
    "vendor_status_id", "vendor_status_title", "vendor_owner_city",
    "vendor_owner_id", "isFreeShipping", "IsAvailable", "IsSaleable",
    "mainAttribute", "categoryTitle", "published", "video_ORIGINAL",
    "promotions",
]
REVIEWS_CREATED_RANGE = ["2018-09-22", "2024-04-09"]


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def download_archive(data_dir: str | Path) -> Path:
    """Download the pinned archive and verify its checksum. Requires curl."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    archive = data_dir / "BaSalam_dataset.rar"
    if not archive.exists():
        subprocess.run(
            ["curl", "-sL", "--retry", "3", "-o", str(archive), ARCHIVE_URL], check=True
        )
    digest = sha256_file(archive)
    if digest != ARCHIVE_SHA256:
        raise ValueError(
            f"Archive checksum mismatch: {digest} != {ARCHIVE_SHA256}. "
            "The upstream dataset changed; update the manifest and decision log."
        )
    return archive


def extract_archive(
    archive: str | Path, out_dir: str | Path, files: list[str] | None = None
) -> list[Path]:
    """Extract the RAR archive. Requires 7-Zip (7zz/7z), unrar, or bsdtar.

    `files` optionally limits extraction to named members (e.g. just the
    products CSV) — useful for partial verification without extracting 3GB.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates = [
        (["7zz", "x", "-y", str(archive), f"-o{out_dir}"], files or []),
        (["7z", "x", "-y", str(archive), f"-o{out_dir}"], files or []),
        (["unrar", "x", "-y", str(archive), str(out_dir) + "/"], files or []),
        (["bsdtar", "-xf", str(archive), "-C", str(out_dir)], files or []),
    ]
    for cmd, only in candidates:
        argv = list(cmd) + list(only)
        try:
            subprocess.run(argv, check=True)
            return sorted(out_dir.glob("*.csv"))
        except (FileNotFoundError, subprocess.CalledProcessError):
            continue
    raise RuntimeError("No RAR extractor found (install 7zip, e.g. static 7zz).")


def _product_dtypes() -> dict:
    return {
        "_id": "string",
        # _score is a deterministic function of the target for nonzero rows —
        # kept in the frame so the audit can VERIFY the leak is present (and
        # the contamination demo can use it); the feature builder structurally
        # forbids it from ever entering a model (features.FORBIDDEN).
        "_score": "Float64",
        "sales_count_week": "Int64",
        "price": "Float64",
        "primaryPrice": "Float64",
        "stock": "Int64",
        "rating_average": "Float64",
        "rating_count": "Int64",
        "preparationDays": "Int64",
        "weight": "Float64",
        "vendor_id": "string",
        "categoryId": "Int64",
        "new_categoryId": "Int64",
        "vendor_score": "Float64",
    }


def products_to_parquet(csv_path: str | Path, parquet_path: str | Path, chunksize: int = 500_000) -> None:
    """Convert the products CSV to typed Parquet in chunks.

    A sidecar records the source hash and output schema. An existing Parquet
    file is reused only when its sidecar, schema, and row count all validate;
    stale, truncated, or legacy cache files are rebuilt atomically.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    csv_path = Path(csv_path)
    parquet_path = Path(parquet_path)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    dtypes = _product_dtypes()
    usecols = list(dtypes)
    source_sha256 = sha256_file(csv_path)
    source_bytes = csv_path.stat().st_size
    metadata_path = parquet_path.with_suffix(parquet_path.suffix + ".meta.json")

    if parquet_path.exists() and metadata_path.exists():
        try:
            cached = json.loads(metadata_path.read_text(encoding="utf-8"))
            parquet_file = pq.ParquetFile(parquet_path)
            valid_cache = (
                cached.get("source_sha256") == source_sha256
                and cached.get("source_bytes") == source_bytes
                and cached.get("columns") == usecols
                and cached.get("rows") == parquet_file.metadata.num_rows
                and parquet_file.schema_arrow.names == usecols
                and parquet_file.metadata.num_rows > 0
            )
            if valid_cache:
                return
        except Exception:
            # Corrupt or legacy caches are not trusted; rebuild below.
            pass

    # Atomic write: build the data and sidecar at temporary paths, then rename
    # only after the conversion completes successfully.
    tmp_path = parquet_path.with_suffix(parquet_path.suffix + ".part")
    tmp_metadata_path = metadata_path.with_suffix(metadata_path.suffix + ".part")
    writer = None
    total_rows = 0
    try:
        for chunk in pd.read_csv(csv_path, usecols=usecols, dtype=dtypes, chunksize=chunksize):
            # pandas preserves source-file order for usecols; explicitly impose
            # the canonical dtype/schema order before writing the Parquet file.
            chunk = chunk[usecols]
            table = pa.Table.from_pandas(chunk, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(tmp_path, table.schema)
            else:
                table = table.cast(writer.schema)
            writer.write_table(table)
            total_rows += len(chunk)
        if writer is None or total_rows == 0:
            raise ValueError(f"no rows read from {csv_path}")
        writer.close()
        writer = None
        tmp_path.replace(parquet_path)
        metadata = {
            "source_sha256": source_sha256,
            "source_bytes": source_bytes,
            "rows": total_rows,
            "columns": usecols,
        }
        tmp_metadata_path.write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        tmp_metadata_path.replace(metadata_path)
    finally:
        if writer is not None:
            writer.close()
        for temporary_path in (tmp_path, tmp_metadata_path):
            if temporary_path.exists():
                temporary_path.unlink()


def load_products(data_dir: str | Path) -> pd.DataFrame:
    parquet = Path(data_dir) / "products.parquet"
    if not parquet.exists():
        raise FileNotFoundError(
            "products.parquet missing. Run scripts/prepare_data.py first "
            "(downloads the pinned archive, verifies SHA-256, converts to Parquet)."
        )
    return pd.read_parquet(parquet)


def verify_score_leak(df: pd.DataFrame) -> float:
    """Fraction of nonzero-sale rows where _score == 500*(y+1)/(y+2) (+-0.01).

    Returns 1.0 when the deterministic leakage is fully present, 0.0 when absent.
    If _score is not in the frame (already stripped), returns -1.0.
    """
    if "_score" not in df.columns:
        return -1.0
    y = pd.to_numeric(df["sales_count_week"], errors="coerce").fillna(0)
    nz = y > 0
    if nz.sum() == 0:
        return 0.0
    score = pd.to_numeric(df.loc[nz, "_score"], errors="coerce")
    expected = 500.0 * (y[nz] + 1) / (y[nz] + 2)
    return float(((score - expected).abs() < 0.01).mean())


def audit_products(df: pd.DataFrame) -> dict:
    """Audit the products frame; results are compared against AUDITED constants."""
    y = pd.to_numeric(df["sales_count_week"], errors="coerce")
    audit = {
        "rows": int(len(df)),
        "zero_share": float((y.fillna(0) == 0).mean()),
        "total_weekly_sales": int(y.fillna(0).sum()),
        "max_weekly_sales": int(y.max()),
        "nonzero_rows": int((y > 0).sum()),
        "distinct_vendors": int(df["vendor_id"].dropna().nunique()),
        "missing_vendor_id": int(df["vendor_id"].isna().sum()),
        "duplicate_product_id": int(df["_id"].duplicated().sum()),
        "score_leak_fraction_nonzero": verify_score_leak(df),
    }
    return audit


def check_audit(audit: dict, tolerance: float = 0.01) -> list[str]:
    """Compare an audit against the pinned expectations; return list of problems.

    The audit dict uses short keys (`rows`, ...) while the pinned constants use
    the manifest keys (`products_rows`, ...); the mapping is explicit here.
    """
    exact = {"rows": "products_rows", "distinct_vendors": "distinct_vendors",
             "missing_vendor_id": "missing_vendor_id", "duplicate_product_id": "duplicate_product_id",
             "total_weekly_sales": "total_weekly_sales", "max_weekly_sales": "max_weekly_sales",
             "nonzero_rows": "nonzero_rows"}
    problems = []
    for akey, pkey in exact.items():
        if pkey in AUDITED and audit.get(akey) != AUDITED[pkey]:
            problems.append(f"{akey}: {audit.get(akey)} != pinned {AUDITED[pkey]}")
    if abs(audit.get("zero_share", 0) - AUDITED["zero_share"]) > tolerance:
        problems.append(f"zero_share: {audit.get('zero_share')} differs from pinned {AUDITED['zero_share']}")
    # The deterministic _score leak MUST be verifiable on the pinned artifact.
    if audit.get("score_leak_fraction_nonzero", -1.0) < 0.99:
        problems.append(
            "score_leak: _score leak not verified (fraction="
            f"{audit.get('score_leak_fraction_nonzero')}); the audit requires _score "
            "in the frame to prove we are on the pinned dataset"
        )
    return problems


def write_manifest(path: str | Path, extra: dict | None = None) -> None:
    manifest = {
        "artifact": "RadeAI/BaSalam_comments_products (Hugging Face)",
        "archive_url": ARCHIVE_URL,
        "archive_sha256": ARCHIVE_SHA256,
        "archive_bytes": ARCHIVE_BYTES,
        "audit": AUDITED,
        "products_columns": PRODUCTS_COLUMNS,
        "reviews_created_range": REVIEWS_CREATED_RANGE,
        "notes": (
            "Static single extract; archive file dates 2024-05-06. Reference week of "
            "sales_count_week is UNDOCUMENTED: this project estimates the recorded "
            "weekly column only and makes no forecasting claims. _score is a "
            "deterministic function of the target for all nonzero-sale rows and is "
            "excluded from all features."
        ),
    }
    if extra:
        manifest.update(extra)
    Path(path).write_text(json.dumps(manifest, indent=2))
