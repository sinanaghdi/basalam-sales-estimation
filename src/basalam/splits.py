"""Split construction: seller-grouped K-fold, random K-fold contrast, manifests.

Guarantees:
- In grouped folds, no vendor appears in both train and test (asserted).
- Every row appears in exactly one test fold.
- Folds are deterministic given (group_col, n_splits, seed).
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd


def group_kfold(
    df: pd.DataFrame, group_col: str, n_splits: int = 5, seed: int = 0
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Deterministic group K-fold: vendors are shuffled and chunked.

    Returns a list of (train_positions, test_positions) aligned to
    df.reset_index(drop=True).
    """
    df = df.reset_index(drop=True)
    # Missing group values (e.g. the 3 rows with no vendor_id in the real
    # data) become their own "<missing>" group so every row lands in exactly
    # one test fold — dropping them would violate full coverage.
    group_str = df[group_col].astype(str).where(df[group_col].notna(), "<missing>")
    groups = np.array(sorted(group_str.unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    folds = np.array_split(groups, n_splits)

    positions = df.index.to_numpy()
    group_values = group_str.to_numpy()
    out = []
    for fold_groups in folds:
        mask = pd.Series(group_values).isin(set(fold_groups.tolist())).to_numpy()
        test_idx = positions[mask]
        train_idx = positions[~mask]
        _assert_disjoint(group_values, train_idx, test_idx, group_col)
        out.append((train_idx, test_idx))
    _assert_full_coverage(df, out)
    return out


def random_kfold(
    df: pd.DataFrame, n_splits: int = 5, seed: int = 0
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Standard shuffled K-fold (the deliberately optimistic contrast regime)."""
    from sklearn.model_selection import KFold

    df = df.reset_index(drop=True)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    positions = df.index.to_numpy()
    return [(positions[tr], positions[te]) for tr, te in kf.split(positions)]


def _assert_disjoint(group_values, train_idx, test_idx, group_col):
    tr_groups = set(group_values[train_idx].tolist())
    te_groups = set(group_values[test_idx].tolist())
    overlap = tr_groups & te_groups
    if overlap:
        raise AssertionError(
            f"Leakage guard: {len(overlap)} {group_col} groups appear in both "
            "train and test (e.g. {sorted(overlap)[:3]})"
        )


def _assert_full_coverage(df, folds):
    all_test = np.concatenate([te for _, te in folds])
    if len(np.unique(all_test)) != len(df):
        raise AssertionError("Split guard: test folds do not cover each row exactly once")


def split_manifest(df: pd.DataFrame, folds, group_col: str | None, seed: int) -> dict:
    """Serializable description of a split (for immutability/reproducibility)."""
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
            gs = df.loc[te, group_col]
            te_groups = sorted(
                set(gs.astype(str).where(gs.notna(), "<missing>").tolist())
            )
            digest = hashlib.sha256(json.dumps(te_groups).encode()).hexdigest()
            fold["test_groups_sha256"] = digest
            fold["n_test_groups"] = len(te_groups)
        manifest["folds"].append(fold)
    return manifest
