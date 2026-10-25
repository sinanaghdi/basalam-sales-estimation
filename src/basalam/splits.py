"""Split construction: seller-grouped K-fold, random K-fold contrast, manifests.

Guarantees:
- In grouped folds, no group appears in both train and test.
- Every row appears in exactly one test fold.
- Missing group IDs are retained as a distinct group, not merged with a
  literal string such as "<missing>".
- Folds are deterministic given (group_col, n_splits, seed).
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd


def _validate_n_splits(n_splits: int, n_rows: int) -> None:
    if not isinstance(n_splits, int) or isinstance(n_splits, bool):
        raise ValueError("n_splits must be an integer")
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    if n_splits > n_rows:
        raise ValueError(f"n_splits ({n_splits}) cannot exceed rows/groups ({n_rows})")


def group_kfold(
    df: pd.DataFrame, group_col: str, n_splits: int = 5, seed: int = 0
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Deterministic group K-fold with one test fold per row."""
    if group_col not in df.columns:
        raise ValueError(f"group column {group_col!r} is missing")
    df = df.reset_index(drop=True)
    # factorize(..., use_na_sentinel=False) gives missing values their own code
    # without colliding with any legitimate string value.
    group_codes, unique_groups = pd.factorize(
        df[group_col], sort=False, use_na_sentinel=False
    )
    _validate_n_splits(n_splits, len(unique_groups))
    groups = np.arange(len(unique_groups))
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    folds = np.array_split(groups, n_splits)

    positions = df.index.to_numpy()
    out = []
    for fold_groups in folds:
        mask = np.isin(group_codes, fold_groups)
        test_idx = positions[mask]
        train_idx = positions[~mask]
        _assert_disjoint(group_codes, train_idx, test_idx, group_col)
        out.append((train_idx, test_idx))
    _assert_full_coverage(df, out)
    return out


def random_kfold(
    df: pd.DataFrame, n_splits: int = 5, seed: int = 0
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Standard shuffled K-fold (the deliberately optimistic contrast regime)."""
    from sklearn.model_selection import KFold

    df = df.reset_index(drop=True)
    _validate_n_splits(n_splits, len(df))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    positions = df.index.to_numpy()
    folds = [(positions[tr], positions[te]) for tr, te in kf.split(positions)]
    _assert_full_coverage(df, folds)
    return folds


def _assert_disjoint(group_values, train_idx, test_idx, group_col):
    tr_groups = set(group_values[train_idx].tolist())
    te_groups = set(group_values[test_idx].tolist())
    overlap = tr_groups & te_groups
    if overlap:
        example = sorted(overlap)[:3]
        raise AssertionError(
            f"Leakage guard: {len(overlap)} {group_col} groups appear in both "
            f"train and test (e.g. {example})"
        )


def _assert_full_coverage(df, folds):
    if not folds:
        raise AssertionError("Split guard: no folds were generated")
    all_test = np.concatenate([te for _, te in folds])
    counts = np.bincount(all_test, minlength=len(df))
    if len(counts) != len(df) or not np.all(counts == 1):
        raise AssertionError(
            "Split guard: test folds must cover every row exactly once"
        )


def split_manifest(df: pd.DataFrame, folds, group_col: str | None, seed: int) -> dict:
    """Serializable split summary with a stable digest of each test partition."""
    df = df.reset_index(drop=True)
    manifest = {
        "group_col": group_col,
        "n_splits": len(folds),
        "seed": seed,
        "folds": [],
    }
    for i, (tr, te) in enumerate(folds):
        fold = {"fold": i, "train_rows": int(len(tr)), "test_rows": int(len(te))}
        if group_col:
            gs = df.iloc[te][group_col]
            _, unique_groups = pd.factorize(gs, sort=False, use_na_sentinel=False)
            group_hashes = pd.util.hash_pandas_object(
                gs, index=False, categorize=True
            ).to_numpy(dtype="<u8", copy=False)
            digest = hashlib.sha256(group_hashes.tobytes()).hexdigest()
            fold["test_groups_sha256"] = digest
            fold["n_test_groups"] = int(len(unique_groups))
        manifest["folds"].append(fold)
    return manifest
