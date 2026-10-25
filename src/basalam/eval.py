"""Evaluation: count-appropriate metrics, rare-positive metrics, strata, macro,
and vendor-cluster bootstrap uncertainty.

Notes:
- Poisson deviance of an all-zero prediction is infinite when any y > 0; the
  runner flags this explicitly instead of silently returning a number.
- Strata are defined on TRUE y: {0, 1-2, 3-10, >10}.
- Category-macro metrics average per-category MAE over categories with at
  least `min_n` products (unweighted macro; choice documented in the plan).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss


def mae(y, pred) -> float:
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(pred))))


def rmse(y, pred) -> float:
    d = np.asarray(y) - np.asarray(pred)
    return float(np.sqrt(np.mean(d**2)))


def poisson_deviance(y, mu) -> float:
    """D = 2*sum[y*log(y/mu) - (y - mu)]; y=0 term is 2*mu. inf if mu<=0 & y>0."""
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    if np.any((mu <= 0) & (y > 0)):
        return float("inf")
    mu = np.maximum(mu, 1e-12)
    terms = np.where(y > 0, y * np.log(np.where(y > 0, y, 1.0) / mu), 0.0) - (y - mu)
    return float(2.0 * np.sum(terms))


def pr_auc(y_binary, p) -> float:
    return float(average_precision_score(np.asarray(y_binary), np.asarray(p)))


def brier(y_binary, p) -> float:
    return float(brier_score_loss(np.asarray(y_binary), np.asarray(p)))


def brier_skill(y_binary, p, base_p=None) -> float:
    """Brier skill score vs. a baseline. >0 = better than baseline.

    `base_p` must be the TRAIN-fold prevalence (one value per row, from the
    same folds that produced `p`) — the prediction-time-valid baseline. If
    omitted, the pooled evaluation prevalence is used, which is NOT valid for
    out-of-fold reporting and is kept only for quick interactive checks.
    """
    yb = np.asarray(y_binary, dtype=float)
    b_model = brier(yb, p)
    if base_p is None:
        prev = yb.mean()
        if prev in (0.0, 1.0):
            return float("nan")
        b_base = brier(yb, np.full_like(yb, prev))
    else:
        b_base = brier(yb, base_p)
    if b_base == 0.0:
        return float("nan")
    return float(1.0 - b_model / b_base)


def volume_stratum(y) -> str:
    y = np.asarray(y)
    return np.where(
        y == 0, "0",
        np.where(y <= 2, "1-2", np.where(y <= 10, "3-10", ">10")),
    )


def stratified_metrics(y, pred) -> dict:
    y = np.asarray(y)
    pred = np.asarray(pred)
    strata = volume_stratum(y)
    out = {}
    for s in ["0", "1-2", "3-10", ">10"]:
        mask = strata == s
        if mask.sum() == 0:
            continue
        out[s] = {
            "n": int(mask.sum()),
            "mae": mae(y[mask], pred[mask]),
            "rmse": rmse(y[mask], pred[mask]),
        }
    return out


def category_macro_metrics(y, pred, categories, min_n: int = 30) -> dict:
    y = np.asarray(y)
    pred = np.asarray(pred)
    cats = np.asarray(categories)
    per_cat = []
    for c in np.unique(cats):
        mask = cats == c
        if mask.sum() >= min_n:
            per_cat.append(mae(y[mask], pred[mask]))
    if not per_cat:
        return {"n_categories": 0, "macro_mae": None}
    return {"n_categories": len(per_cat), "macro_mae": float(np.mean(per_cat))}


def vendor_cluster_bootstrap_ci(
    y, pred, vendors, stat_fn=None, n_boot: int = 200, seed: int = 0, agg: str = "mean"
) -> dict:
    """Cluster bootstrap over vendors (resample vendors, keep their rows).

    For MAE (the default) this uses an O(n) fast path: per-vendor absolute-
    error sums and counts are precomputed once, and each replicate just sums
    the resampled vendors' totals. Missing vendor IDs (NaN/None) form their
    own cluster via pandas factorize. Arbitrary `stat_fn` falls back to explicit
    row resampling (slow; used only in tests to verify the fast path).
    """
    stat_fn = stat_fn or mae
    y = np.asarray(y)
    pred = np.asarray(pred)
    # pandas `string` dtype NA survives .astype(str) (conversion is a no-op)
    # and factorize would emit -1 codes — fill first, then stringify.
    codes, _ = pd.factorize(pd.Series(vendors).fillna("<missing>").astype(str))
    rng = np.random.default_rng(seed)
    n_v = codes.max() + 1
    if stat_fn is mae:
        abs_err = np.abs(y - pred)
        sums = np.bincount(codes, weights=abs_err, minlength=n_v)
        counts = np.bincount(codes, minlength=n_v)
        stats = []
        for _ in range(n_boot):
            take = rng.integers(0, n_v, size=n_v)
            stats.append(sums[take].sum() / counts[take].sum())
        point = sums.sum() / counts.sum()
    else:
        # Generic (slow) path: explicit row resampling per replicate.
        idx_by_v = [np.where(codes == v)[0] for v in range(n_v)]
        stats = []
        for _ in range(n_boot):
            take = rng.integers(0, n_v, size=n_v)
            idx = np.concatenate([idx_by_v[v] for v in take])
            stats.append(stat_fn(y[idx], pred[idx]))
        point = stat_fn(y, pred)
    stats = np.array(stats)
    return {
        "point": point,
        "ci_low": float(np.percentile(stats, 2.5)),
        "ci_high": float(np.percentile(stats, 97.5)),
        "n_boot": n_boot,
    }


def jsonable(obj):
    """Convert numpy/pandas scalars and inf/nan to JSON-safe values."""
    if isinstance(obj, dict):
        return {k: jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        if np.isinf(f):
            return "inf"
        if np.isnan(f):
            return "nan"
        return f
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    return obj
