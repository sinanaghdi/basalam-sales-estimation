"""Experiment runner.

Usage:
    python -m basalam.run configs/experiments/smoke.yaml

Runs the full pipeline on synthetic or real data:
  1. audit (target distribution, `_score` leakage check) and fail hard on
     forbidden columns;
  2. seller-grouped K-fold and random K-fold regimes with identical models,
     baselines, and tuning;
  3. optional contamination namespace (deliberately including `_score`),
     written to a separate output directory so it can never be confused with
     primary results;
  4. JSON results with seeds, config hash, git commit, and library versions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import data as data_mod
from . import eval as ev
from . import features as ft
from . import models as md
from . import splits as sp
from .synthetic import generate_synthetic

TARGET = "sales_count_week"
GROUP = "vendor_id"
CAT = "new_categoryId"


def _load_config(path: str | Path) -> dict:
    import yaml

    return yaml.safe_load(Path(path).read_text())


def _config_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return "unknown (not a git repo)"


def get_data(cfg: dict) -> pd.DataFrame:
    if cfg["data"]["source"] == "synthetic":
        s = cfg["data"]["synthetic"]
        return generate_synthetic(**s)
    elif cfg["data"]["source"] == "real":
        from .data import load_products  # requires pyarrow

        df = load_products(cfg["data"].get("data_dir", "data"))
        return df
    raise ValueError(f"unknown data source: {cfg['data']['source']}")


def _evaluate_regime(
    df: pd.DataFrame, folds, cfg: dict, contamination: bool = False
) -> dict:
    seed = cfg.get("seed", 42)
    params = cfg.get("models", {}).get("params", {})
    n_boot = cfg.get("uncertainty", {}).get("cluster_bootstrap", 50)

    model_names = ["gbm_poisson", "hurdle", "global_median", "category_median", "global_positive_mean"]
    oof = {name: np.zeros(len(df)) for name in model_names}
    oof_p = np.zeros(len(df))  # hurdle P(y>0)
    oof_base_p = np.zeros(len(df))  # per-row TRAIN-fold prevalence (Brier baseline)

    for train_idx, test_idx in folds:
        train = df.iloc[train_idx]
        test = df.iloc[test_idx]
        y_train = train[TARGET].to_numpy()
        y_test = test[TARGET].to_numpy()

        # Vendor/category stats learned on TRAIN rows only (fold-internal)
        vstats = ft.fit_vendor_stats(train)
        cstats = ft.fit_category_stats(train, cat_col=CAT)
        X_train = ft.build_features(train, vstats, cstats)
        X_test = ft.build_features(test, vstats, cstats)
        if contamination:
            # DELIBERATE leakage: _score is a deterministic function of the target
            X_train = pd.concat(
                [X_train.reset_index(drop=True), train[["_score"]].reset_index(drop=True)], axis=1
            )
            X_test = pd.concat(
                [X_test.reset_index(drop=True), test[["_score"]].reset_index(drop=True)], axis=1
            )

        oof["gbm_poisson"][test_idx] = md.fit_predict_gbm_poisson(
            X_train, y_train, X_test, params
        )
        hurdle = md.fit_hurdle(X_train, y_train, params)
        oof["hurdle"][test_idx] = md.hurdle_predict(hurdle, X_test)
        oof_p[test_idx] = hurdle["clf"].predict_proba(X_test)[:, 1]

        oof["global_median"][test_idx] = md.global_median(y_train)
        lookup = md.category_median_lookup(train, y_train, CAT)
        oof["category_median"][test_idx] = md.predict_category_median(test, lookup, CAT)
        oof["global_positive_mean"][test_idx] = md.global_positive_mean(y_train)
        # TRAIN-fold prevalence, constant on this fold's test rows: the
        # prediction-time-valid baseline for the hurdle's Brier skill score.
        prev = float((y_train > 0).mean())
        oof_base_p[test_idx] = prev
        if prev in (0.0, 1.0):
            oof_base_p[test_idx] = np.clip(prev, 1e-6, 1 - 1e-6)

    y = df[TARGET].to_numpy()
    ybin = (y > 0).astype(int)
    vendors = df[GROUP].to_numpy()
    cats = df[CAT].to_numpy()

    results = {"n_rows": int(len(df)), "nonzero_share": float(ybin.mean())}
    for name, pred in oof.items():
        entry = {
            "mae": ev.mae(y, pred),
            "rmse": ev.rmse(y, pred),
            "poisson_deviance": ev.poisson_deviance(y, pred),
        }
        if name == "hurdle":
            entry["pr_auc"] = ev.pr_auc(ybin, oof_p)
            entry["brier"] = ev.brier(ybin, oof_p)
            entry["brier_skill"] = ev.brier_skill(ybin, oof_p, base_p=oof_base_p)
        entry["strata"] = ev.stratified_metrics(y, pred)
        entry["category_macro"] = ev.category_macro_metrics(y, pred, cats)
        entry["mae_ci"] = ev.vendor_cluster_bootstrap_ci(
            y, pred, vendors, stat_fn=ev.mae, n_boot=n_boot, seed=seed
        )
        results[name] = entry
    return results


def run_experiment(cfg: dict) -> dict:
    seed = cfg.get("seed", 42)
    n_splits = cfg.get("splits", {}).get("n_splits", 5)
    regimes = cfg.get("splits", {}).get("regimes", ["grouped_vendor", "random"])

    df = get_data(cfg).reset_index(drop=True)

    # --- Phase 1 audit: runs BEFORE any modeling and BLOCKS it on real data ---
    audit = data_mod.audit_products(df)
    if cfg["data"]["source"] == "real":
        problems = data_mod.check_audit(audit)
        if problems:
            raise RuntimeError(
                "Real-data audit failed; modeling is blocked:\n  - "
                + "\n  - ".join(problems)
            )
        audit["problems"] = []
    if audit["score_leak_fraction_nonzero"] >= 0.99:
        audit["score_leak"] = (
            "DETECTED: _score is a deterministic function of the target for "
            "nonzero rows; excluded from all primary features"
        )

    results = {
        "experiment": cfg.get("name", "unnamed"),
        "audit": audit,
        "regimes": {},
    }

    for regime in regimes:
        if regime == "grouped_vendor":
            folds = sp.group_kfold(df, GROUP, n_splits=n_splits, seed=seed)
            manifest = sp.split_manifest(df, folds, GROUP, seed)
        elif regime == "grouped_category":
            folds = sp.group_kfold(df, CAT, n_splits=n_splits, seed=seed)
            manifest = sp.split_manifest(df, folds, CAT, seed)
        elif regime == "random":
            folds = sp.random_kfold(df, n_splits=n_splits, seed=seed)
            manifest = sp.split_manifest(df, folds, None, seed)
        else:
            raise ValueError(f"unknown regime: {regime}")
        results["regimes"][regime] = {
            "split_manifest": manifest,
            **_evaluate_regime(df, folds, cfg),
        }

    if cfg.get("contamination_demo", False):
        folds = sp.group_kfold(df, GROUP, n_splits=n_splits, seed=seed)
        results["contamination_demo"] = {
            "note": (
                "DELIBERATE contamination (_score included). Never compare these "
                "numbers with primary results; this namespace exists to quantify "
                "the leakage trap."
            ),
            **_evaluate_regime(df, folds, cfg, contamination=True),
        }

    results["metadata"] = {
        "seed": seed,
        "config_hash": _config_hash(cfg),
        "git_commit": _git_commit(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "data_fingerprint": _data_fingerprint(df),
        "library_versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "sklearn": __import__("sklearn").__version__,
            "lightgbm": __import__("lightgbm").__version__,
        },
    }
    return results


def _data_fingerprint(df: pd.DataFrame) -> str:
    """Return a stable fingerprint of target and group values.

    Hash canonical pandas value hashes rather than raw object-array memory:
    object-array bytes can encode process-specific pointer values. Row order is
    intentionally significant because this pins the exact experiment input.
    """
    missing_token = "<missing>"
    target = pd.to_numeric(df[TARGET], errors="coerce").astype("float64")
    groups = df[GROUP].astype("string").fillna(missing_token)

    h = hashlib.sha256()
    h.update(pd.util.hash_pandas_object(target, index=False).to_numpy(dtype="<u8").tobytes())
    h.update(pd.util.hash_pandas_object(groups, index=False).to_numpy(dtype="<u8").tobytes())
    h.update(str(len(df)).encode("ascii"))
    return h.hexdigest()[:16]


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="path to experiment YAML config")
    args = parser.parse_args(argv)
    cfg = _load_config(args.config)

    results = run_experiment(cfg)

    out_dir = Path(cfg.get("output_dir", "outputs/experiments")) / cfg.get(
        "name", "unnamed"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "results.json"
    out_file.write_text(json.dumps(ev.jsonable(results), indent=2))

    # Console summary
    print(f"\nExperiment: {results['experiment']}  ->  {out_file}")
    first_regime = next(iter(results["regimes"]))
    print(f"rows={results['regimes'][first_regime]['n_rows']}  nonzero_share={results['regimes'][first_regime]['nonzero_share']:.4f}")
    for regime, res in results["regimes"].items():
        print(f"\n[{regime}]")
        for model in ("global_median", "category_median", "global_positive_mean",
                      "gbm_poisson", "hurdle"):
            if model in res:
                m = res[model]
                print(
                    f"  {model:24s} MAE={m['mae']:.4f} "
                    f"RMSE={m['rmse']:.4f} "
                    f"dev={m['poisson_deviance'] if m['poisson_deviance'] == 'inf' else format(m['poisson_deviance'], '.1f')}"
                )
    return 0


if __name__ == "__main__":
    sys.exit(main())
