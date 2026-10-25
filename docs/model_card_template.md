# Model Card (template) — fill before release

Adapted from Mitchell et al. (2019), Model Cards for Model Reporting.

## Model details
- Person/organization developing: 
- Date and version: 
- License: (code; the dataset is NOT redistributed)
- Intended use: estimation of the RECORDED `sales_count_week` column of a static
  BaSalam extract (May 2024 file dates). Not forecasting. Not causal.

## Intended use / out-of-scope uses
- Out-of-scope: future-week forecasting; other marketplaces without revalidation;
  pricing/causal decisions; individual-seller ranking as a decision system.

## Factors and evaluation data
- Groups analyzed: vendor (seller), category, sales-volume stratum (0 / 1-2 /
  3-10 / >10), vendor size.
- Data: pinned archive (see manifests/data_manifest.json), SHA-256 verified.

## Metrics
- Pooled and per-stratum MAE, RMSE, Poisson deviance; PR-AUC, Brier skill for
  P(sale ≥ 1); vendor-cluster bootstrap 95% CIs; category-macro MAE.

## Evaluation results
- (fill from results.json; report grouped and random regimes side by side,
  clearly labeled)

## Ethical considerations / limitations
- Static snapshot; unknown reference week; extreme zero inflation (96.8%);
  BaSalam data republication restricted; association only — no causal claims;
  Persian-text features may introduce category-unequal measurement error.
