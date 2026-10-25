# Estimating Weekly Product Sales in the BaSalam Marketplace
### A Leakage-Aware Evaluation of Explainable Models across Sellers and Categories

Reproducible codebase for the research design described in `PROJECT_PLAN.md`.
The project estimates the **recorded** `sales_count_week` column of a static
BaSalam marketplace extract (2.41M products, 3.39M reviews, 88,822 sellers) —
**contemporaneous estimation, not forecasting** — under seller-grouped and
category-grouped evaluation.

## Quickstart

```bash
make install          # pip install -e ".[dev]"
make test             # unit + leakage-guard tests on synthetic data (no download)
make smoke            # end-to-end pipeline on synthetic data (~1.5 min)
make install-real     # OPTIONAL: adds pyarrow, for real-data work
make prepare-data     # OPTIONAL: download pinned archive (434MB), verify SHA-256, convert to Parquet
                      #   (needs 7zz/7z/unrar/bsdtar on PATH for RAR extraction)
make audit-real       # full-data audit + grouped experiment (needs prepare-data)
```

A GitHub Actions workflow is supplied (`.github/workflows/ci.yml`) to run
the tests and smoke pipeline on pushes and pull requests. Check the
[Actions page](https://github.com/sinanaghdi/basalam-sales-estimation/actions)
for the current run status; workflow status is time-dependent and should not be
inferred from this README.

## What this repository guarantees

- **Pinned data**: the archive SHA-256, audited statistics, and column schema
  are recorded in `manifests/data_manifest.json`; the loader fails hard if the
  upstream dataset changes.
- **Leakage guards**: the target, the deterministic target-proxy `_score`, and
  identifiers are structurally excluded from features (`features.FORBIDDEN`);
  grouped folds assert vendor-disjointness; vendor/category aggregates are
  learned inside training folds with global fallbacks for unseen groups.
- **Honest evaluation**: baselines are prediction-time valid (training global /
  category medians, positive global mean); metrics are count-appropriate
  (Poisson deviance, MAE/RMSE) plus rare-positive metrics for the hurdle
  stage (PR-AUC, Brier skill); uncertainty is vendor-cluster bootstrap.
- **Determinism**: same seed + config + data hash → identical metrics within
  float tolerance (tested). Results record seeds, config hash, git commit,
  library versions, and split manifests.

## Key facts about the data (audited 2026-10-24)

| Property | Value |
|---|---|
| Products / reviews | 2,411,358 / 3,393,574 |
| Zero-sales share (target) | 96.81% |
| Sellers / category groups | 88,822 / 816 (`new_categoryId`) |
| `_score` | deterministic function of the target for all 76,942 nonzero rows — excluded |
| Reference week | undocumented — estimation only, no forecasting claims |
| Reviews date range | 2018-09-22 … 2024-04-09 |

## Repository layout

```
configs/experiments/*.yaml   # experiment configs (smoke, audit-real)
src/basalam/data.py          # download, SHA-256 pin, Parquet, audit
src/basalam/synthetic.py     # synthetic fixture mimicking schema + leakage
src/basalam/features.py      # fold-internal features + leakage guards
src/basalam/splits.py        # group K-fold, random contrast, manifests
src/basalam/models.py        # Poisson GBM, hurdle, valid baselines
src/basalam/eval.py          # count + rare-positive metrics, cluster bootstrap
src/basalam/run.py           # experiment runner
tests/                       # leakage guards, metric invariants, determinism
docs/decision_log.md         # every methodological decision
docs/model_card_template.md
scripts/prepare_data.py
```

## Data governance

The raw dataset is **never committed**. The Hugging Face card labels it MIT,
but the data originates from BaSalam's public API; BaSalam's terms of use may
restrict republication. This repository publishes code and derived statistics
only. The LICENSE covers code.

## Current status

The repository contains the data-audit gate, leakage-aware feature pipeline,
grouped and random evaluation regimes, baselines, count and rare-positive
metrics, vendor-cluster bootstrap intervals, synthetic smoke data, and CI
tests. The full research plan still includes additional work such as paired
bootstrap comparisons, multi-seed robustness analysis, feature ablations,
SHAP analysis, and completion of the model card. These are not complete merely
because the smoke pipeline passes.

Before presenting real-data results as research findings, run the pinned
real-data audit and document the exact configuration, commit, data fingerprint,
and limitations. The reference week is undocumented, so the target must not be
described as future-sales forecasting.

See `PROJECT_PLAN.md` for milestones, hypotheses, and acceptance criteria.
