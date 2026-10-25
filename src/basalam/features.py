"""Feature construction with leakage-safe, fold-internal vendor statistics.

Rules (see docs/decision_log.md):
- FORBIDDEN columns (target, `_score`, identifiers) can never enter the matrix.
- Vendor-level aggregate features are learned on TRAIN rows only; test vendors
  unseen in training fall back to the TRAIN global value.
- Category aggregates are unavailable for unseen categories; the same global
  fallback applies.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# The target, the deterministic target proxy, and identifiers/join keys.
FORBIDDEN = {
    "sales_count_week",  # target
    "_score",             # deterministic function of target for nonzero rows
    "_id", "productId", "id",
    "vendor_id", "vendor_identifier", "vendor_owner_id", "user_id",
    "navigation_id", "hashId",
}

BASE_NUMERIC = [
    "price", "primaryPrice", "rating_average", "rating_count",
    "stock", "vendor_score", "weight", "preparationDays",
]


def assert_no_forbidden(columns) -> None:
    bad = sorted(set(columns) & FORBIDDEN)
    if bad:
        raise ValueError(f"Leakage guard: forbidden columns in feature matrix: {bad}")


def fit_vendor_stats(train_df: pd.DataFrame) -> dict:
    """Learn vendor-level feature aggregates from TRAIN rows only."""
    stats = {
        "vendor_price_mean": train_df.groupby("vendor_id")["price"].mean().to_dict(),
        "vendor_n_products": train_df["vendor_id"].value_counts().to_dict(),
        "global_price_mean": float(train_df["price"].mean()),
    }
    return stats


def _vendor_transform(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    price_map = stats["vendor_price_mean"]
    count_map = stats["vendor_n_products"]
    glob = stats["global_price_mean"]
    out = pd.DataFrame(index=df.index)
    # .astype("float64") strips arrow-backed dtypes (double[pyarrow]), which
    # LightGBM rejects — real-data frames load from Parquet as arrow-backed.
    out["vendor_price_mean"] = (
        df["vendor_id"].map(price_map).fillna(glob).astype("float64")
    )
    out["vendor_n_products"] = (
        df["vendor_id"].map(count_map).fillna(0).astype("float64")
    )
    return out


def fit_category_stats(train_df: pd.DataFrame, cat_col: str = "new_categoryId") -> dict:
    return {
        "category_price_mean": train_df.groupby(cat_col)["price"].mean().to_dict(),
        "global_price_mean": float(train_df["price"].mean()),
        "cat_col": cat_col,
    }


def _category_transform(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    cat_col = stats["cat_col"]
    out = pd.DataFrame(index=df.index)
    out["category_price_mean"] = (
        df[cat_col].map(stats["category_price_mean"])
        .fillna(stats["global_price_mean"])
        .astype("float64")
    )
    return out


def build_features(
    df: pd.DataFrame,
    vendor_stats: dict | None = None,
    category_stats: dict | None = None,
) -> pd.DataFrame:
    """Build the feature matrix. Vendor/category stats MUST come from the
    training folds; passing None disables those aggregate features."""
    parts = []
    numeric = [c for c in BASE_NUMERIC if c in df.columns]
    if numeric:
        # Cast nullable pandas dtypes (Int64/Float64 with pd.NA) to plain
        # float64 — LightGBM does not accept pd.NA sentinels.
        base = df[numeric].astype("float64").reset_index(drop=True)
        parts.append(base)
    else:
        base = pd.DataFrame(index=np.arange(len(df)))

    if "price" in base.columns:
        parts.append(
            pd.DataFrame({"log_price": np.log1p(base["price"].clip(lower=0))})
        )
    if "rating_count" in base.columns:
        parts.append(
            pd.DataFrame({"log_rating_count": np.log1p(base["rating_count"].clip(lower=0))})
        )
    if vendor_stats is not None and "vendor_id" in df.columns:
        parts.append(_vendor_transform(df, vendor_stats).reset_index(drop=True))
    if category_stats is not None:
        parts.append(_category_transform(df, category_stats).reset_index(drop=True))

    X = pd.concat(parts, axis=1)
    assert_no_forbidden(X.columns)
    return X
