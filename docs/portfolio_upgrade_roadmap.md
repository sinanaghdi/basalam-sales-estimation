# Portfolio Upgrade Roadmap — BaSalam Market Intelligence

## Purpose

Turn this repository into a defensible portfolio case study for Data Analyst and
Applied Machine Learning roles. The goal is not to maximize model complexity.
The goal is to demonstrate data quality work, sound validation, useful business
questions, reproducible analysis, and clear communication.

## Current capabilities

- Pinned source archive and data audit checks.
- Synthetic data for CI and end-to-end smoke testing without distributing raw data.
- Leakage guards for the target, `_score`, identifiers, and seller identifiers.
- Fold-internal seller/category aggregates and grouped validation.
- Baselines, Poisson GBM, hurdle model, count metrics, rare-positive metrics,
  and vendor-cluster bootstrap intervals.
- Experiment metadata and split manifests.

## Portfolio workstreams

### 1. Reliability and data contracts

- Keep the real-data audit as a hard gate before fitting.
- Make the experiment fingerprint stable for nullable and ordinary pandas dtypes.
- Test missing category and seller identifiers, malformed inputs, and invalid
  prediction arrays.
- Validate an existing Parquet artifact before treating it as a completed conversion.
- Keep CI independent of private or large raw data.

**Acceptance criteria**
- Unit tests cover nullable target and group columns.
- Repeated runs on the same values produce the same fingerprint.
- A changed target or group value changes the fingerprint.
- Missing categories do not crash macro metrics.
- CI tests and synthetic smoke pass on a clean environment.

### 2. Statistical evaluation that withstands review

- Compare grouped-seller and random-split results with the same data and model
  settings, while explicitly treating random split as an optimistic contrast.
- Add paired seller-cluster bootstrap intervals for differences between regimes
  and between candidate models.
- Run a predeclared multi-seed profile; report the distribution across seeds.
- Include baseline comparisons, category macro metrics, target-volume strata,
  and probability calibration diagnostics.
- Keep synthetic positive-control results clearly separate from findings on real data.

**Acceptance criteria**
- Every comparison names its metric, direction, sample, seed set, and uncertainty.
- No claim of statistical support is made from overlapping independent intervals.
- Real-data results are traceable to commit, config hash, data fingerprint,
  and library versions.

### 3. Business-facing analysis

Answer questions that can be supported by the available snapshot:

- How concentrated are listed products and recorded sales across sellers and categories?
- Which categories have the highest zero-sales share and the largest uncertainty?
- How do price, ratings, stock, preparation time, and seller-level aggregates
  relate to recorded sales, without claiming causation?
- Where does the model fail: zero-sales products, low-volume products, and
  higher-volume products?
- Does performance differ between seen-like and unseen seller evaluation settings?

Do not claim future sales forecasting: the reference week is undocumented.
Do not present observational feature importance as causal impact.

**Deliverables**
- A reproducible EDA notebook with data dictionary and explicit caveats.
- Exported figures generated from code, not manually edited.
- A short executive summary with findings, uncertainty, and suggested next analyses.
- A model card describing intended use, limitations, data risks, and evaluation.

### 4. Explainability and robustness

- Compare base product features against seller/category aggregates.
- Perform feature ablation under grouped and random regimes.
- Add SHAP only after the validation pipeline and model comparisons are stable.
- Test sensitivity to category taxonomy and relevant product-availability filters.
- Measure runtime and peak memory for the real-data workflow.

**Acceptance criteria**
- Every ablation uses identical folds and comparable configurations.
- Feature importance is not described as causal.
- Runtime and memory claims include environment details.

### 5. Professional presentation

- Keep README concise, accurate, and reproducible.
- Add an architecture diagram, experiment matrix, results table generated from
  machine-readable output, and limitations section.
- Provide exact setup commands for Windows PowerShell and Unix-like shells.
- Include sample output from synthetic data only; never fabricate real-data results.
- Keep raw marketplace data and credentials out of Git history.

## Recommended execution order

1. Reliability tests and CI.
2. Real-data audit and reproducible baseline run.
3. Paired statistical comparisons and multi-seed robustness.
4. EDA and business-facing findings.
5. Feature ablation and explainability.
6. Model card, figures, and final portfolio narrative.

## Definition of done

The project is ready to present when another developer can install it, run tests,
reproduce the synthetic smoke experiment, understand exactly how real-data access
works, inspect a traceable results artifact, and see which conclusions are
supported versus unresolved. Passing CI alone is not evidence that the research
hypotheses have been confirmed.
