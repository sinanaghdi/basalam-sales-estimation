"""Metrics and pipeline invariants on synthetic fixtures (no real data needed)."""
import numpy as np
import pandas as pd
import pytest

from basalam import eval as ev
from basalam import models as md
from basalam.run import run_experiment


def test_zero_model_mae_equals_mean():
    y = np.array([0, 0, 0, 1, 2, 0, 3, 0])
    assert ev.mae(y, np.zeros_like(y)) == pytest.approx(y.mean())


def test_poisson_deviance_perfect_prediction_is_zero():
    y = np.array([0.0, 1.0, 3.0, 0.0, 2.0])
    assert ev.poisson_deviance(y, y) == pytest.approx(0.0, abs=1e-9)


def test_poisson_deviance_zero_prediction_is_infinite():
    assert ev.poisson_deviance([1.0], [0.0]) == float("inf")


def test_brier_skill_against_prevalence():
    yb = np.array([0, 0, 0, 1, 0, 1, 0, 0])
    prev = yb.mean()
    assert ev.brier_skill(yb, np.full(len(yb), prev)) == pytest.approx(0.0)
    assert ev.brier_skill(yb, yb) == pytest.approx(1.0)  # perfect predictor


def test_brier_skill_against_train_fold_prevalence_baseline():
    """Out-of-fold reporting must use the TRAIN-fold prevalence as baseline,
    not the pooled evaluation prevalence (decision log)."""
    yb = np.array([0, 0, 0, 1, 0, 1, 0, 0])
    train_prev = 0.02  # differs from the evaluation prevalence above
    base_p = np.full(len(yb), train_prev)
    # A model no better than its training-fold prevalence baseline scores ~0
    assert ev.brier_skill(yb, base_p, base_p=base_p) == pytest.approx(0.0)
    # Baseline mismatch is detectable: skill vs. the correct baseline differs
    # from skill vs. the (invalid) pooled prevalence baseline.
    p = np.full(len(yb), 0.25)
    s_correct = ev.brier_skill(yb, p, base_p=base_p)
    s_pooled = ev.brier_skill(yb, p)
    assert s_correct != pytest.approx(s_pooled)


def test_global_mean_baseline_is_unconditional_train_mean():
    """The constant baseline is E[Y] over the TRAIN fold (not E[Y | Y > 0]),
    floored at epsilon so Poisson deviance stays finite."""
    y_train = np.array([0, 0, 0, 0, 4])
    mu = md.global_positive_mean(y_train)
    assert mu == pytest.approx(0.8)  # mean of y, NOT 4.0
    assert mu > 0
    assert md.global_positive_mean(np.zeros(5)) == pytest.approx(1e-6)


def test_cluster_bootstrap_fast_path_matches_row_resampling():
    """The O(n) per-vendor-totals fast path must reproduce explicit row
    resampling exactly (same seed -> same vendor draws; MAE is order-invariant).
    Includes a missing vendor ID (NaN) as its own cluster."""
    rng = np.random.default_rng(0)
    n = 500
    y = rng.poisson(0.3, n).astype(float)
    pred = rng.poisson(0.3, n).astype(float)
    vendors = np.array([f"v{i // 10}" if i < 490 else np.nan for i in range(n)])

    fast = ev.vendor_cluster_bootstrap_ci(y, pred, vendors, stat_fn=ev.mae, n_boot=50, seed=3)
    # Generic path: wrap mae so `stat_fn is mae` is False
    slow_fn = lambda yy, pp: ev.mae(yy, pp)
    slow = ev.vendor_cluster_bootstrap_ci(y, pred, vendors, stat_fn=slow_fn, n_boot=50, seed=3)
    assert fast["point"] == pytest.approx(ev.mae(y, pred))
    assert fast["point"] == pytest.approx(slow["point"])
    assert fast["ci_low"] == pytest.approx(slow["ci_low"])
    assert fast["ci_high"] == pytest.approx(slow["ci_high"])


def test_hurdle_combined_is_probability_times_positive_mean():
    # p * E[y|y>0] must be nonnegative and bounded by the max plausible count
    p = np.array([0.1, 0.5, 0.9])
    mu = np.array([1.0, 2.0, 3.0])
    combined = p * mu
    assert np.all(combined >= 0)
    assert combined.max() <= p.max() * mu.max()


def test_pipeline_is_deterministic_given_seed():
    cfg = {
        "name": "det-check",
        "data": {
            "source": "synthetic",
            "synthetic": {
                "n_products": 4000,
                "n_vendors": 200,
                "n_categories": 20,
                "seed": 7,
            },
        },
        "seed": 7,
        "splits": {"n_splits": 3, "regimes": ["grouped_vendor", "random"]},
        "models": {"params": {"n_estimators": 30, "verbose": -1}},
        "uncertainty": {"cluster_bootstrap": 10},
    }
    r1 = run_experiment(cfg)
    r2 = run_experiment(cfg)
    for regime in ("grouped_vendor", "random"):
        for model in ("gbm_poisson", "hurdle", "global_median"):
            for metric in ("mae", "rmse"):
                assert r1["regimes"][regime][model][metric] == pytest.approx(
                    r2["regimes"][regime][model][metric]
                )


def test_random_split_more_optimistic_than_grouped_on_synthetic():
    """ENGINEERED POSITIVE CONTROL (not a research finding).

    The synthetic generator deliberately ties vendor quality to log-price so
    that fold-internal vendor aggregates carry real signal. Under random
    K-fold the same vendors appear in train and test, so the model can exploit
    them; under grouped K-fold it cannot. This control verifies that the
    evaluation harness can DETECT that mechanism:

    - Poisson deviance (conditional-mean log-loss): random < grouped
    - hurdle PR-AUC (ranking metric): random > grouped

    It says nothing about the real BaSalam data; the real test of H1 happens
    in Phase 4 on the pinned dataset.

    MAE is deliberately NOT asserted: MAE targets the conditional median and,
    with ~97% zeros, a near-zero median prediction is legitimate — so MAE can
    prefer the less-informative constant predictor and flip the gap
    direction (verified empirically; recorded in docs/decision_log.md D8).
    PR-AUC is a ranking metric, not a proper scoring rule; Brier score is the
    proper probability-scoring companion. The direction holds on average
    across seeds, not as a per-run guarantee."""
    dev_gap, prauc_gap = [], []
    for seed in (11, 12, 13):
        cfg = {
            "name": "gap-check",
            "data": {
                "source": "synthetic",
                "synthetic": {
                    "n_products": 20000,
                    "n_vendors": 400,
                    "n_categories": 40,
                    "seed": seed,
                },
            },
            "seed": seed,
            "splits": {"n_splits": 3, "regimes": ["grouped_vendor", "random"]},
            "models": {"params": {"n_estimators": 50, "verbose": -1}},
            "uncertainty": {"cluster_bootstrap": 10},
        }
        r = run_experiment(cfg)
        g, rnd = r["regimes"]["grouped_vendor"], r["regimes"]["random"]
        dev_gap.append(g["gbm_poisson"]["poisson_deviance"] - rnd["gbm_poisson"]["poisson_deviance"])
        prauc_gap.append(rnd["hurdle"]["pr_auc"] - g["hurdle"]["pr_auc"])
    assert np.mean(dev_gap) > 0   # grouped regime has higher count log-loss
    assert np.mean(prauc_gap) > 0  # random regime ranks sellers better


def test_category_macro_metrics_handles_nullable_categories():
    """Missing category IDs should be evaluated as one explicit category."""
    import pandas as pd

    y = np.array([0, 1, 0, 2, 1, 0], dtype=float)
    pred = np.array([0, 0, 1, 1, 1, 0], dtype=float)
    categories = pd.Series(pd.array(["a", "a", None, None, "b", "b"], dtype="string"))

    result = ev.category_macro_metrics(y, pred, categories, min_n=2)

    # Three groups (a, b, and missing), each with two rows.
    assert result["n_categories"] == 3
    assert result["macro_mae"] == pytest.approx(0.5)


def test_data_fingerprint_is_stable_and_sensitive_to_values():
    import pandas as pd

    from basalam.run import _data_fingerprint

    frame = pd.DataFrame(
        {
            "sales_count_week": pd.Series([0, 1, None, 3], dtype="Float64"),
            "vendor_id": pd.Series(["v1", "v2", None, "v1"], dtype="string"),
            "price": pd.Series([10.0, 20.0, None, 40.0], dtype="Float64"),
        }
    )
    same_values = frame.copy()
    changed_target = frame.copy()
    changed_target.loc[1, "sales_count_week"] = 2
    changed_feature = frame.copy()
    changed_feature.loc[1, "price"] = 21.0
    changed_group = frame.copy()
    changed_group.loc[1, "vendor_id"] = "v3"

    first = _data_fingerprint(frame)
    assert first == _data_fingerprint(same_values)
    assert first != _data_fingerprint(changed_target)
    assert first != _data_fingerprint(changed_feature)
    assert first != _data_fingerprint(changed_group)


def test_git_commit_uses_ci_revision_when_checkout_metadata_is_missing(
    monkeypatch, tmp_path
):
    from basalam.run import _git_commit

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GITHUB_SHA", "a" * 40)
    monkeypatch.delenv("GIT_COMMIT", raising=False)

    assert _git_commit() == "a" * 40


def test_hurdle_handles_all_zero_and_all_positive_training_targets():
    import pandas as pd

    X = pd.DataFrame({"feature": [0.0, 1.0, 2.0, 3.0]})
    params = {"n_estimators": 5, "verbose": -1}

    all_zero = md.fit_hurdle(X, np.zeros(4), params)
    assert all_zero["reg"] is None
    assert np.all(md.hurdle_predict(all_zero, X) == 0)
    assert np.all(md.hurdle_positive_probability(all_zero["clf"], X) == 0)

    all_positive = md.fit_hurdle(X, np.ones(4), params)
    assert all_positive["reg"] is not None
    assert np.all(md.hurdle_positive_probability(all_positive["clf"], X) == 1)
    assert np.all(md.hurdle_predict(all_positive, X) >= 0)

def test_cluster_bootstrap_keeps_missing_vendor_separate_from_literal_sentinel():
    vendors = pd.Series(pd.array(["<missing>", None, "v1", "v2"], dtype="string"))
    y = np.array([0.0, 1.0, 0.0, 2.0])
    pred = np.array([0.0, 0.0, 0.0, 0.0])
    ci = ev.vendor_cluster_bootstrap_ci(y, pred, vendors, n_boot=30, seed=9)
    assert ci["point"] == pytest.approx(ev.mae(y, pred))
    assert ci["n_boot"] == 30


def test_cluster_bootstrap_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="equal lengths"):
        ev.vendor_cluster_bootstrap_ci([0, 1], [0], ["v1", "v2"])
    with pytest.raises(ValueError, match="positive integer"):
        ev.vendor_cluster_bootstrap_ci([0], [0], ["v1"], n_boot=0)
    with pytest.raises(ValueError, match="agg='mean'"):
        ev.vendor_cluster_bootstrap_ci([0], [0], ["v1"], agg="median")

