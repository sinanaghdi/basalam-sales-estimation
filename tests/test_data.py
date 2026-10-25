"""Data-intake tests: chunked Parquet conversion, atomicity, manifest."""
import numpy as np
import pandas as pd
import pytest

from basalam import data as data_mod


def _sample_frame(n: int = 250, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    y = rng.poisson(0.05, n)
    df = pd.DataFrame(
        {
            "_id": [f"p{i}" for i in range(n)],
            "_score": 500.0 * (y + 1) / (y + 2),
            "sales_count_week": y,
            "price": rng.lognormal(11.5, 0.7, n),
            "primaryPrice": rng.lognormal(11.5, 0.7, n),
            "stock": rng.poisson(50, n),
            "rating_average": rng.uniform(0, 5, n),
            "rating_count": rng.poisson(3, n),
            "preparationDays": rng.poisson(2, n),
            "weight": rng.uniform(0, 5, n),
            "vendor_id": [f"v{i % 25}" for i in range(n)],
            "categoryId": rng.integers(0, 8, n),
            "new_categoryId": rng.integers(0, 8, n),
            "vendor_score": rng.uniform(0, 1, n),
        }
    )
    # Nullable values: missing prices and vendor_ids, as in the real data
    df.loc[3, "price"] = None
    df.loc[7, "vendor_id"] = None
    df.loc[11, "sales_count_week"] = None
    return df


def test_products_to_parquet_chunked_and_schema(tmp_path):
    """Multiple chunks, nullable values, and `_score` preservation."""
    df = _sample_frame()
    csv_path = tmp_path / "sample.csv"
    df.to_csv(csv_path, index=False)
    parquet_path = tmp_path / "sample.parquet"
    data_mod.products_to_parquet(csv_path, parquet_path, chunksize=100)  # 3 chunks
    assert parquet_path.exists()
    assert not (tmp_path / "sample.parquet.part").exists()
    back = pd.read_parquet(parquet_path)
    assert len(back) == len(df)
    assert list(back.columns) == list(df.columns)
    # `_score` must survive (the audit needs it) and stay leaky
    assert data_mod.verify_score_leak(back) == pytest.approx(1.0)
    # Nullable values preserved
    assert pd.isna(back.loc[3, "price"])
    assert pd.isna(back.loc[7, "vendor_id"])


def test_products_to_parquet_atomic_and_idempotent(tmp_path):
    df = _sample_frame()
    csv_path = tmp_path / "sample.csv"
    df.to_csv(csv_path, index=False)
    parquet_path = tmp_path / "sample.parquet"
    data_mod.products_to_parquet(csv_path, parquet_path, chunksize=100)
    first_mtime = parquet_path.stat().st_mtime_ns
    # Existing parquet is not rebuilt (conversion happens once)
    data_mod.products_to_parquet(csv_path, parquet_path, chunksize=100)
    assert parquet_path.stat().st_mtime_ns == first_mtime
    # A failed conversion must not leave a complete-looking file behind:
    # corrupt the CSV mid-way and convert to a NEW path; the .part temp file
    # is removed and no final parquet is produced.
    bad_csv = tmp_path / "bad.csv"
    lines = csv_path.read_text().splitlines()
    bad_csv.write_text("\n".join(lines[:50]) + "\n,garbage\n")
    bad_parquet = tmp_path / "bad.parquet"
    with pytest.raises(Exception):
        data_mod.products_to_parquet(bad_csv, bad_parquet, chunksize=100)
    assert not bad_parquet.exists()
    assert not (tmp_path / "bad.parquet.part").exists()


def test_build_features_strips_arrow_dtypes(tmp_path):
    """Frames loaded from real Parquet are arrow-backed (double[pyarrow]);
    build_features must emit plain float64 columns that LightGBM accepts.
    Regression test for the real-data Phase 1 run."""
    pytest.importorskip("pyarrow")
    import numpy as np

    from basalam import features as ft

    df = pd.DataFrame(
        {
            "price": np.array([10.0, 20.0, 30.0, 40.0], dtype="float64"),
            "vendor_id": ["a", "a", "b", "b"],
            "new_categoryId": pd.array([1, 1, 2, 2], dtype="Int64"),
            "sales_count_week": pd.array([0, 1, 0, 2], dtype="Int64"),
        }
    )
    # Convert to arrow-backed dtypes, as pd.read_parquet returns them
    df = df.convert_dtypes(dtype_backend="pyarrow")
    vs, cs = ft.fit_vendor_stats(df.iloc[:2]), ft.fit_category_stats(df.iloc[:2])
    X = ft.build_features(df, vs, cs)
    assert all(str(dt) in ("float64", "int64") for dt in X.dtypes), list(X.dtypes)


def test_manifest_roundtrip(tmp_path):
    path = tmp_path / "manifest.json"
    data_mod.write_manifest(path)
    import json

    m = json.loads(path.read_text())
    assert m["archive_sha256"] == data_mod.ARCHIVE_SHA256
    assert m["audit"]["products_rows"] == data_mod.AUDITED["products_rows"]

def test_products_to_parquet_rebuilds_when_source_changes_without_row_count_change(tmp_path):
    df = _sample_frame(20)
    csv_path = tmp_path / "source.csv"
    df.to_csv(csv_path, index=False)
    parquet_path = tmp_path / "products.parquet"

    data_mod.products_to_parquet(csv_path, parquet_path, chunksize=7)
    first = pd.read_parquet(parquet_path)
    original_price = first.loc[0, "price"]

    # Keep the row count and schema fixed; a source fingerprint must still
    # invalidate the cache when the underlying values change.
    df.loc[0, "price"] = float(original_price) + 123.0
    df.to_csv(csv_path, index=False)
    data_mod.products_to_parquet(csv_path, parquet_path, chunksize=7)

    rebuilt = pd.read_parquet(parquet_path)
    assert rebuilt.loc[0, "price"] == pytest.approx(float(original_price) + 123.0)
    metadata_path = parquet_path.with_suffix(".parquet.meta.json")
    assert metadata_path.exists()

