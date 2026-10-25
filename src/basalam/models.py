"""Models: count-objective GBM, two-part hurdle, and prediction-time-valid baselines.

Design notes (see PROJECT_PLAN.md):
- Baselines must be computable for UNSEEN vendors/categories: category medians
  fall back to the training global median; vendor medians are never used as a
  baseline for grouped evaluation (unseen vendors have no history).
- The all-zero predictor has infinite Poisson deviance when any y > 0; a
  positive global-mean baseline is included so deviance is always finite.
- Hurdle prediction is the combined unconditional estimate p * E[y | y > 0].
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.dummy import DummyClassifier

DEFAULT_PARAMS = {
    "n_estimators": 50,
    "num_leaves": 15,
    "min_child_samples": 20,
    "learning_rate": 0.1,
    "verbose": -1,
    "random_state": 0,
}


def fit_gbm_poisson(X: pd.DataFrame, y: np.ndarray, params: dict | None = None):
    m = LGBMRegressor(objective="poisson", **{**DEFAULT_PARAMS, **(params or {})})
    m.fit(X, y)
    return m


def fit_hurdle(X: pd.DataFrame, y: np.ndarray, params: dict | None = None):
    """Two-part model: P(y > 0) via binary GBM; E[y | y > 0] via Poisson GBM.

    Degenerate folds with only zero or only positive targets use a constant
    classifier instead of asking LightGBM to fit a one-class classification
    problem. The positive-count regressor is fitted only when positive rows
    exist.
    """
    p = {**DEFAULT_PARAMS, **(params or {})}
    y = np.asarray(y)
    y_binary = (y > 0).astype(int)
    if np.unique(y_binary).size == 2:
        clf = LGBMClassifier(objective="binary", **p)
    else:
        clf = DummyClassifier(strategy="constant", constant=int(y_binary[0]))
    clf.fit(X, y_binary)

    pos = y > 0
    reg = None
    if pos.any():
        reg = LGBMRegressor(objective="poisson", **p)
        X_positive = X.loc[pos] if hasattr(X, "loc") else X[pos]
        reg.fit(X_positive, y[pos])
    return {"clf": clf, "reg": reg}


def hurdle_positive_probability(clf, X: pd.DataFrame) -> np.ndarray:
    """Return P(y > 0), including classifiers trained on a single class."""
    classes = np.asarray(getattr(clf, "classes_", []))
    if not np.any(classes == 1):
        return np.zeros(len(X), dtype=float)
    probabilities = clf.predict_proba(X)
    positive_index = int(np.flatnonzero(classes == 1)[0])
    return probabilities[:, positive_index]


def hurdle_predict(models: dict, X: pd.DataFrame) -> np.ndarray:
    p = hurdle_positive_probability(models["clf"], X)
    if models["reg"] is None:
        return np.zeros(len(X))
    mu_pos = models["reg"].predict(X)
    return p * np.maximum(mu_pos, 0.0)


def fit_predict_gbm_poisson(X_train, y_train, X_test, params=None):
    m = fit_gbm_poisson(X_train, y_train, params)
    return np.maximum(m.predict(X_test), 0.0)


def zero_model(y_train: np.ndarray) -> float:
    return 0.0


def global_median(y_train: np.ndarray) -> float:
    return float(np.median(y_train))


def global_positive_mean(y_train: np.ndarray) -> float:
    """TRAIN-fold UNCONDITIONAL mean, floored at epsilon.

    The all-zero predictor has infinite Poisson deviance when any y > 0; a
    positive constant keeps deviance finite. This is the training-fold mean
    E[Y] (not E[Y | Y > 0]) — the prediction-time-valid constant baseline."""
    mu = float(np.mean(y_train)) if len(y_train) else 0.0
    return max(mu, 1e-6)


def category_median_lookup(
    train_df: pd.DataFrame, y_train: np.ndarray, cat_col: str
) -> dict:
    """Category medians from TRAIN, with global-median fallback for unseen categories."""
    s = pd.Series(np.asarray(y_train), index=train_df.index)
    med = s.groupby(train_df[cat_col]).median().to_dict()
    med["__global__"] = float(np.median(y_train))
    return med


def predict_category_median(test_df: pd.DataFrame, lookup: dict, cat_col: str) -> np.ndarray:
    glob = lookup["__global__"]
    return test_df[cat_col].map(lookup).fillna(glob).to_numpy(dtype=float)
