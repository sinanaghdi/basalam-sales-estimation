# Reproducible Project Plan — BaSalam Weekly Sales Estimation

**Title:** Estimating Weekly Product Sales in the BaSalam Marketplace: A Leakage-Aware Evaluation of Explainable Models across Sellers and Categories

**Repository:** `basalam-sales-estimation/` (this skeleton — runnable, tested, not yet published to GitHub)
**Design source:** BaSalam_Literature_Review.md, §5 (final recommendation)
**Verified on:** 2026-10-24

---

## 1. Project Overview

A minimal, reproducible GitHub project that implements the leakage-aware evaluation
design from the literature review: contemporaneous estimation of the recorded
`sales_count_week` column from a static BaSalam extract (2.41M products, 3.39M
reviews, 88,822 sellers, 96.81% zeros), with seller- and category-grouped
evaluation, prediction-time-valid baselines, and a contamination demo that
quantifies the `_score` leak.

**Estimation, not forecasting.** Single static extract; the reference week is
undocumented. All claims are about the recorded column only.

**Design decisions carried into code (see `docs/decision_log.md`):**

| Decision | Rationale |
|---|---|
| Data pinned by SHA-256 | Upstream (HF) changes must fail loudly, not silently invalidate comparisons |
| `_score` + target excluded from features | `_score = 500·(y+1)/(y+2)` verified for all 76,942 nonzero rows |
| Audit BEFORE modeling | Leakage and data changes must block experiments, not surface as "bad results" |
| Fold-internal preprocessing | Vendor/category aggregates learned on train folds only; global fallback for unseen groups |
| Grouped K-fold by vendor; random as contrast | Measures out-of-seller generalization vs interpolation among seen sellers |
| Contamination demo in separate namespace | Deliberately-leaky run quantifies the leak; never mixed with primary results |
| Vendor-cluster bootstrap | Fold-level CV understates uncertainty under grouped splits (Kapoor & Narayanan 2023) |

---

## 2. Phase Plan (0–8)

### Phase 0 — Bootstrap & smoke pipeline ✅ DONE
Skeleton with src-layout, synthetic fixture reproducing the real leakage
properties (96.8% zeros, deterministic `_score` leak, 88.8k-like vendor
structure), runnable end to end on synthetic data, tested in CI.

**Acceptance criteria:** `pytest` green (22 tests); `make smoke` produces
`outputs/experiments/smoke/results.json` with both regimes and the contamination
demo in its own namespace; audit functions verified against the real extracted
CSV (leak fraction 1.0 on a 400k-row sample); the RAR extractor verified
against the pinned archive (byte-identical products CSV).

**Phase 0 empirical results (smoke = ENGINEERED POSITIVE CONTROL, synthetic
data, seed 42 — not a finding about BaSalam):** the harness detects the H1
mechanism when it is deliberately planted: random K-fold Poisson deviance 5,553
vs grouped 10,864. MAE flips the direction (0.104 vs 0.093): MAE targets the
conditional median and, with ~97% zeros, a near-zero median prediction is
legitimate, so MAE can prefer the less-informative constant predictor (D8:
proper scoring metrics are primary for hypotheses; PR-AUC is a ranking metric,
Brier score is the proper probability-scoring companion). The contamination demo
quantifies the `_score` trap: deviance 231 (with leak) vs 10,864 (without) —
a 47x apparent improvement from leakage alone, reported only in its isolated
namespace. The pre-split vendor-aggregate experiment for H3 is FUTURE WORK
(the current demo deliberately adds only `_score`).

### Phase 1 — Real data intake & audit ✅ DONE (commit 79831c7, 2026-10-24)
`make prepare-data` downloads the pinned archive (434 MB), verifies SHA-256,
converts products to Parquet, writes/updates `manifests/data_manifest.json`.
`make audit-real` runs the full audit (row counts, zero share, `_score` leak
present, vendor counts) BEFORE any modeling, then the grouped experiment.

**Executed:** prepare-data verified end-to-end (archive SHA-256 match;
extraction verified by FULL-file SHA-256: products CSV
`38f5988…359fad`, identical to the reference extraction; chunked Parquet
`f0da5f47…1627e`, 2,411,358 rows in 5 row groups). The audit gate PASSED —
every pinned expectation reproduced exactly (see D10). Four real-data
defects found and regression-tested: arrow-backed dtypes rejected by LightGBM,
`string`-NA crashing the bootstrap factorize, NaN vendor groups violating fold
coverage, slow per-row fold masks. Observed RSS during the experiment ~1.6 GB
(process snapshot; true peak not measured — a measurement item for Phase 4).

**First real results (single seed, seller-disjoint folds, `_score` excluded):**

| Model | Poisson deviance | MAE | Notes |
|---|---|---|---|
| Train-mean baseline | 766,960 | 0.0848 | prediction-time-valid constant |
| Poisson-GBM | 676,580 | 0.1144 | |
| Hurdle | 603,428 | 0.1018 | PR-AUC 0.186 (prevalence 0.032), Brier skill +0.053 |

These are SINGLE-SEED, SELLER-DISJOINT PILOT results with `_score` excluded —
descriptive only, not formal support for any hypothesis (the planned paired
vendor-cluster uncertainty analysis comes in Phase 4). The hurdle's PR-AUC
0.186 and Brier skill +0.053 are above the train-fold prevalence baseline, the
H4 DIRECTION, but formal H4 support requires the uncertainty analysis. On MAE
the zero-median baselines win — the documented D8 caveat. Strata MAE grows
with volume (0.07 / 0.88 / 3.29 / 13.2 across {0, 1-2, 3-10, >10});
vendor-cluster bootstrap CI for hurdle MAE: [0.099, 0.105]. No random-regime
contrast, no tuning, feature set v0.

**Run provenance:** the results JSON records commit `08562cec`, but the run
included the Phase 1 fixes that were committed immediately afterwards as
`79831c7` — use `79831c7` (or later) as the reproduction point; the recorded
commit predates the fixes it actually ran.

**Milestone (GitHub issue):** "Real-data audit reproduces pinned expectations." ✅
**Acceptance:** audit metrics in `results.json` match `expected` within tolerance;
local tests pass (hosted CI pending — the workflow has not yet run on GitHub
Actions). ✅

### Phase 2 — Feature engineering
Current features are a subset (price, log price, stock, preparation, ratings,
vendor aggregates, category aggregates). Extend per the review: discounts,
shipping/availability flags, review-derived aggregates (count, recency with
cutoff), status/vendor-activity flags. All learned fold-internally with global
fallbacks; `_score`, target, identifiers structurally forbidden (enforced by tests).

**Milestone:** "Feature set v1 with fold-internal vendor/category aggregates."

### Phase 3 — Model set
Poisson-GBM (implemented), hurdle (implemented: Poisson GBM gate + rate model
on positive-only, rate forced ≥ ε), plus: Poisson/NB GLM, RF Poisson/Tweedie,
XGBoost Poisson. All behind a uniform fit/predict interface.

**Milestone:** "All review models implementable; benchmark table populated."

### Phase 4 — Evaluation protocol
Implemented: GroupKFold vendor (disjointness tested), random contrast,
count metrics (MAE/RMSE/Poisson deviance with ε-clamped predictions),
rare-positive metrics (PR-AUC, Brier skill over training prevalence),
strata (0 / 1–2 / 3–10 / >10), category-macro, vendor-cluster bootstrap.
Extend per review: LOGO-category stress test, seller-matching inner folds,
taxonomy sensitivity (`categoryId` vs `new_categoryId`), `IsSaleable`/`IsAvailable`
filtering, multiple seeds.

**Milestone:** "Full metric battery incl. grouped-vs-random gap and stress tests."

### Phase 5 — Explainability
TreeSHAP per model (folds, SHAP-vs-features matrix). Rank-correlate vendor- vs
non-vendor feature importances across regimes; compare aggregate vs no-aggregate
feature sets (aggregate-feature leakage diagnosis). Deliver one global
explanation figure + one vendor-level example.

**Milestone:** "SHAP explanations reproduce across seeds and regimes; H5 tested."

### Phase 6 — Robustness & reporting
Seed variance, single-baseline dominance plots, LEAVE-ONE-OUT-style masking
(fit without vendor features → does grouped error increase?), reporting per
stratum and per seller-size decile. Model card (`docs/model_card_template.md`)
completed; limitations section covers: static snapshot, undocumented reference
week, no forecasting claims, BaSalam data redistribution restrictions.

**Milestone:** "All robustness checks from §5 pass or documented as threats."

### Phase 7 — Paper writing
Sections per review structure: Intro → Related work → Dataset audit → Method →
Results (grouped vs random, H1–H5) → SHAP → Limitations → Repro. Literature
(31 verified papers; 18 PDFs in `papers/`) cited from the review's clean
bibliography.

**Milestone:** "Draft paper with figures generated from `results.json`."

### Phase 8 — Release
GitHub repository published with: pinned data manifest (no raw data), CI green
on synthetic smoke, decision log current, model card filled, reproduction
instructions in README. `pip install -e .` + `make smoke` works on a clean
machine.

---

## 3. Experiment Matrix

| Experiment | Config | Regimes | Data |
|---|---|---|---|
| Smoke (CI) | `configs/experiments/smoke.yaml` | grouped_vendor, random | synthetic |
| Contamination demo | `smoke.yaml` + `contamination_demo: true` | grouped_vendor | synthetic (with `_score`) |
| Real audit | `configs/experiments/audit_real.yaml` | grouped_vendor | real, pinned |
| Full grid | Phase 4 configs | grouped_vendor, random, LOGO-category, taxonomy ×2 | real, pinned |

Models × features grid: {Poisson-GBM, hurdle, GLM, RF, XGBoost} × {base,
+vendor aggregates, +category aggregates, +all} — run under grouped and random
regimes; vendor-feature ablation answers H3/H4.

---

## 4. Hypotheses (H1–H5), tests, and statistical analysis plan

Hypotheses as stated in the research design (literature review §5):

| # | Hypothesis | Test | Evidence source |
|---|---|---|---|
| H1 | Random-split error underestimates grouped-split error | Compare metrics across regimes | results.json per-regime comparison |
| H2 | The gap is larger for GBMs than for GLMs | Regime × model interaction | results.json |
| H3 | Vendor-aggregate features computed pre-split inflate apparent skill | Fit with/without vendor aggregates; pre-split vs fold-internal pipelines | FUTURE WORK: pre-split contamination namespace + ablation |
| H4 | P(sale ≥ 1) is estimated better than the prevalence baseline | Hurdle PR-AUC and Brier skill vs train-fold prevalence baseline | results.json (hurdle entries) |
| H5 | SHAP importance of vendor features drops from random to grouped | Rank-correlation of SHAP vectors | Phase 5 outputs |

### Statistical analysis plan

- **Primary endpoint per hypothesis:** H1/H2 — Poisson deviance (count
  log-loss) per model per regime; H3 — deviance gap between pre-split and
  fold-internal feature pipelines; H4 — hurdle Brier skill score against the
  train-fold prevalence baseline (PR-AUC reported as the ranking companion);
  H5 — Spearman rank correlation of vendor-feature SHAP importances between
  regimes.
- **Uncertainty for differences (not just per-model point values):** for each
  hypothesis contrast (e.g., grouped minus random deviance per model), compute
  PAIRED vendor-cluster bootstrap intervals — resample vendors once per
  replicate and carry BOTH sides of the contrast through the same resample, so
  the interval reflects the difference, not two independent intervals.
- **Seeds:** 5 seeds per regime × model for the full profile (3 for pilot);
  report median and spread across seeds; the seed dimension never shares
  resamples with the bootstrap dimension.
- **Decision criteria:** a hypothesis contrast is supported when the paired
  cluster-bootstrap 95% interval excludes zero in the predicted direction
  consistently across seeds (≥ 4 of 5); otherwise reported as not supported.
  No claims of confirmation — estimation only.
- **Profiles:** smoke (synthetic, CI, minutes), pilot (real data, 1 seed,
  3-fold, small models, sanity-checks the full-data path), full (real data,
  5 seeds × 5 folds, full model grid, the numbers reported in the paper).

---

## 5. Testing Strategy

- **Unit:** metrics (deviance, Brier skill, cluster bootstrap CI), feature builder, splitters.
- **Leakage guards:** grouped folds vendor-disjoint; forbidden columns rejected;
  train-fold medians used for category baselines; contamination runs isolated in
  their own namespace; determinism (same seed/config → same metrics within tolerance).
- **Integration:** smoke pipeline on synthetic data (CI, every push).
- **Real data:** only via `make prepare-data` + `make audit-real`; audit gates
  modeling.

---

## 6. Scale & Environment Notes

- 2.41M products / 2.78GB of CSVs. The CSV→Parquet conversion is chunked
  (500k rows per chunk), so conversion memory is bounded by chunk size; the
  audit and modeling then load the FULL converted Parquet frame (observed
  ~1.6 GB RSS during the Phase 1 experiment; peak not measured — a
  Phase 4 measurement item).
- RAR extraction requires an external tool on PATH: `7zz`/`7z` (recommended;
  static 7zz binaries exist), `unrar`, or `bsdtar`. The extractor was verified
  against the pinned archive (byte-identical products CSV).
- Real-data extras: `pip install -e ".[dev,real]"` (adds pyarrow).
- Raw data never committed; `.gitignore` enforces `data/` and `outputs/` exclusion.
- Dependency versions tested in this session: Python 3.14, numpy 2.5.4,
  pandas 3.1.0, scikit-learn / LightGBM as recorded in results.json metadata.
  A pinned constraints file is a Phase 8 release item.

---

## 7. Key References (verified URLs)

- Dataset (pinned): https://huggingface.co/datasets/RadeAI/BaSalam_comments_products
- Kapoor & Narayanan 2023, leakage & grouped CV: https://doi.org/10.1016/j.patter.2023.100804
- Roberts et al. 2017, cross-validation strategies: https://doi.org/10.1111/ecog.02881
- Makridakis et al. 2018, M4: https://doi.org/10.1371/journal.pone.0194889
- Syntetos & Boylan 2004 (intermittent demand): https://doi.org/10.1016/j.ijforecast.2004.10.001
- Ferreira et al. 2016 (Zara analytics): https://doi.org/10.1287/msom.2015.0561
- TreeSHAP: https://arxiv.org/abs/1802.03888 ; SHAP: https://arxiv.org/abs/1705.07874

The full 31-paper literature review and verified PDF library are separate
deliverables from this repository (not bundled in the ZIP).

## 8. Publishing Checklist (Phase 8)

1. Create GitHub repo `basalam-sales-estimation`; push skeleton.
2. Confirm CI green on synthetic smoke.
3. Add topics: `marketplace`, `sales-estimation`, `leakage`, `grouped-cv`, `shap`.
4. Add `CITATION.cff` referencing the paper once published.
5. Verify README quickstart on a clean clone.
6. Confirm no raw/derived BaSalam data in repo history (`.gitignore` + audit).
7. Link the HF dataset card in README with the redistribution caveat.

---

## 9. Definition of Done (whole project)

- [x] Phase 0 skeleton runnable + tested
- [ ] Phases 1–8 milestones closed as GitHub issues
- [ ] All H1–H5 tested with results in `results.json`
- [ ] Model card filled; decision log current through Phase 8
- [ ] Paper draft complete; reproducible from repo tag

---

## 10. Threats & Mitigations

| Threat | Mitigation |
|---|---|
| Undocumented reference week | Disclosed in paper + model card; estimation-only claims |
| Static extract (no temporal validation) | Disclosed; no forecasting claims |
| BaSalam ToS on data republication | Raw data never committed; only SHA-pinned manifest |
| Upstream dataset changes | SHA-256 pin + hard-fail audit |
| Extreme zero inflation | Hurdle model + count-appropriate metrics; PR-AUC/Brier for stage 1 |
| Cloudflare-blocked publisher PDFs (18/31 obtained) | Remaining reads via library access; marked [BROWSER] in review, never fabricated |
