"""Leakage guards and fold-internal preprocessing tests."""
import numpy as np
import pandas as pd
import pytest

from basalam import data as data_mod
from basalam import eval as ev
from basalam import features as ft
from basalam import splits as sp
from basalam.synthetic import generate_synthetic


def test_forbidden_columns_rejected():
    with pytest.raises(ValueError, match="Leakage guard"):
        ft.assert_no_forbidden(["price", "_score"])
    with pytest.raises(ValueError, match="Leakage guard"):
        ft.assert_no_forbidden(["sales_count_week"])


def test_features_never_contain_forbidden():
    df = generate_synthetic(n_products=2000, n_vendors=100, seed=1)
    X = ft.build_features(df, ft.fit_vendor_stats(df))
    assert set(X.columns) & ft.FORBIDDEN == set()


def test_vendor_stats_from_train_only_with_global_fallback():
    df = generate_synthetic(n_products=4000, n_vendors=200, seed=2)
    train = df.iloc[:3000]
    test = df.iloc[3000:]
    stats = ft.fit_vendor_stats(train)
    X_test = ft.build_features(test, stats)
    unseen = ~test["vendor_id"].isin(stats["vendor_price_mean"])
    # Unseen vendors must fall back to the TRAIN global mean, not their own stats
    assert np.allclose(
        X_test.loc[unseen.values, "vendor_price_mean"],
        stats["global_price_mean"],
    )


def test_grouped_folds_are_vendor_disjoint():
    df = generate_synthetic(n_products=3000, n_vendors=150, seed=3)
    folds = sp.group_kfold(df, "vendor_id", n_splits=5, seed=0)
    for tr, te in folds:
        tr_groups = set(df.iloc[tr]["vendor_id"])
        te_groups = set(df.iloc[te]["vendor_id"])
        assert not (tr_groups & te_groups)


def test_audit_detects_score_leakage():
    df = generate_synthetic(n_products=3000, n_vendors=100, seed=4)
    leak = data_mod.verify_score_leak(df)
    assert leak == pytest.approx(1.0)  # every nonzero row leaks via _score


def test_cluster_bootstrap_handles_pandas_string_na():
    """Genuine pd.NA inside a pandas `string` Series (the real vendor_id
    dtype): NA must survive as its own cluster, not crash factorize.
    Regression test for the Phase 1 real-data run."""
    vendors = pd.Series(pd.array(["v0", "v0", None, "v1"], dtype="string"))
    y = np.array([0.0, 1.0, 0.0, 0.0])
    pred = np.array([0.1, 0.1, 0.1, 0.1])
    ci = ev.vendor_cluster_bootstrap_ci(y, pred, vendors, n_boot=20, seed=0)
    assert ci["ci_low"] <= ci["point"] <= ci["ci_high"]
    # Missing vendor rows form a cluster: point estimate is the overall MAE
    assert ci["point"] == pytest.approx(ev.mae(y, pred))


def test_group_kfold_covers_rows_with_missing_group():
    """Rows with missing vendor_id (3 in the real data) must land in exactly
    one test fold; groups stay disjoint. Regression test for the Phase 1
    real-data run."""
    df = pd.DataFrame(
        {
            "vendor_id": ["a", "a", "b", None, None, "c"],
            "x": range(6),
        }
    )
    folds = sp.group_kfold(df, "vendor_id", n_splits=2, seed=0)
    all_test = np.concatenate([te for _, te in folds])
    assert len(np.unique(all_test)) == len(df)  # each row exactly once
    # Disjointness: no vendor (incl. <missing>) in both train and test
    for tr, te in folds:
        tr_v = set(pd.Series(df.iloc[tr]["vendor_id"]).fillna("<missing>"))
        te_v = set(pd.Series(df.iloc[te]["vendor_id"]).fillna("<missing>"))
        assert not (tr_v & te_v)


def test_audit_flags_missing_score():
    df = generate_synthetic(n_products=1000, n_vendors=50, seed=5)
    df_no_score = df.drop(columns=["_score"])
    assert data_mod.verify_score_leak(df_no_score) == -1.0


def test_real_data_audit_blocks_modeling():
    """A synthetic frame must fail the pinned real-data audit — the gate's
    detection layer (the runner raising on it is covered end-to-end below)."""
    df = generate_synthetic(n_products=2000, n_vendors=100, seed=6)
    audit = data_mod.audit_products(df)
    problems = data_mod.check_audit(audit)
    assert problems, "synthetic frame must fail the pinned real-data audit"
    assert any("rows" in p or "zero_share" in p for p in problems)


def test_audit_gate_in_run_experiment(tmp_path, monkeypatch):
    """End-to-end gate test: run_experiment with source=real on a frame that
    fails the pinned audit must raise BEFORE fitting anything."""
    import basalam.run as run_mod

    df = generate_synthetic(n_products=2000, n_vendors=100, seed=7)
    parquet_dir = tmp_path / "data"
    parquet_dir.mkdir()
    df.to_parquet(parquet_dir / "products.parquet", index=False)

    cfg = {
        "name": "audit-gate-e2e",
        "data": {"source": "real", "data_dir": str(parquet_dir)},
        "seed": 1,
        "splits": {"n_splits": 3, "regimes": ["grouped_vendor"]},
        "models": {"params": {"n_estimators": 5, "verbose": -1}},
        "uncertainty": {"cluster_bootstrap": 5},
    }
    # The synthetic frame fails the pinned real-data expectations -> gate fires.
    with pytest.raises(RuntimeError, match="audit failed"):
        run_mod.run_experiment(cfg)


def test_check_audit_uses_pinned_keys():
    """The audit dict uses short keys; the pinned constants use manifest keys.
    A row-count mismatch must be detected (regression for the key mismatch)."""
    good = {
        "rows": data_mod.AUDITED["products_rows"],
        "distinct_vendors": data_mod.AUDITED["distinct_vendors"],
        "missing_vendor_id": data_mod.AUDITED["missing_vendor_id"],
        "duplicate_product_id": data_mod.AUDITED["duplicate_product_id"],
        "zero_share": data_mod.AUDITED["zero_share"],
        "total_weekly_sales": data_mod.AUDITED["total_weekly_sales"],
        "max_weekly_sales": data_mod.AUDITED["max_weekly_sales"],
        "nonzero_rows": data_mod.AUDITED["nonzero_rows"],
        "score_leak_fraction_nonzero": 1.0,
    }
    assert data_mod.check_audit(good) == []
    bad = dict(good, rows=good["rows"] - 1, score_leak_fraction_nonzero=-1.0)
    problems = data_mod.check_audit(bad)
    assert any("rows" in p for p in problems)
    assert any("score_leak" in p for p in problems)
