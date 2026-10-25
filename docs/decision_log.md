# Decision Log

Architectural and methodological decisions, newest last. Every change to the
data pin, feature rules, or evaluation protocol must be recorded here.

## D1 — 2026-10-24: Data artifact pinned
- Source: `RadeAI/BaSalam_comments_products` (Hugging Face), single RAR archive.
- SHA-256: `12e79a5bfe8322d9a084e327ecb79bc64ea5d232ce157daf0dd06cbad5ab0555` (434,741,588 bytes).
- Audited: 2,411,358 products; 3,393,574 reviews; 96.81% zero-sales share;
  88,822 distinct vendors; 3 missing `vendor_id`; 0 duplicate `_id`.
- The reference week of `sales_count_week` is UNDOCUMENTED. The project estimates
  the recorded column only. No forecasting claims, no claims about "typical"
  weeks. Archive file dates (2024-05-06) are consistent with, but do not prove,
  the collection window.
- Raw data is never committed (BaSalam ToS may restrict republication; the HF
  "MIT" label does not override the source platform's terms).

## D2 — 2026-10-24: `_score` excluded — deterministic target leakage
- For all 76,942 nonzero-sale rows, `_score` = 500*(y+1)/(y+2) within 0.01.
- `_score`, the target, and identifiers are in `features.FORBIDDEN`; the audit
  verifies the leak is present (so we know we are on the pinned dataset) and the
  feature builder rejects them structurally.
- The contamination demo (deliberately including `_score`) lives in a separate
  output namespace and is never compared with primary results.

## D3 — 2026-10-24: Estimation, not forecasting
- Single static extract: contemporaneous estimation of the recorded weekly
  column. Temporal validation impossible; reviews span 2018-09..2024-04 and may
  feed features with an explicit cutoff, but the target is observed once.

## D4 — 2026-10-24: Baselines must be prediction-time valid
- Category-median baselines use TRAIN-fold category medians with a TRAIN global
  fallback for unseen categories; vendor medians are never used as grouped-eval
  baselines (unseen vendors have no history).
- The all-zero predictor has infinite Poisson deviance when any y > 0; a positive
  global-mean baseline is included so deviance is always finite.

## D5 — 2026-10-24: Fold-internal preprocessing
- Vendor/category aggregate features are learned on training folds only; unseen
  groups fall back to training global values. Tests enforce this.

## D6 — 2026-10-24: Random vs grouped framing
- Random K-fold is reported as a contrast regime measuring interpolation among
  seen vendors; grouped K-fold measures out-of-vendor generalization. The gap
  between them is a change of question, not automatically leakage. Vendor-level
  aggregate features computed on the full data before splitting IS leakage
  (tested separately in the contamination namespace).

## D7 — 2026-10-24: Reproducibility contract
- Same seed + same config + same data hash => identical metrics within float
  tolerance (tested); not bitwise identity across platforms.
- Every result JSON records: seed, config hash, git commit, library versions,
  split manifest (fold group checksums), timestamp.

## D8 — 2026-10-24: Hypothesis contrasts are evaluated on proper scoring metrics
- MAE targets the CONDITIONAL MEDIAN; Poisson deviance targets the conditional
  mean. With ~97% zeros, a near-zero median prediction is legitimate, so MAE
  can prefer the less-informative constant predictor — on the engineered
  synthetic control it flipped the grouped-vs-random gap direction. Consequence
  for the paper: Poisson deviance and Brier skill are the primary metrics for
  H1–H4; PR-AUC is a ranking metric (reported as the companion for the hurdle
  stage); MAE/RMSE are reported as descriptives with this caveat stated
  explicitly. The synthetic experiment is an ENGINEERED POSITIVE CONTROL: it
  verifies the harness can detect a deliberately planted vendor signal and
  supports no conclusions about BaSalam itself.

## D9 — 2026-10-24: Review fixes (verification pass)
- The audit now BLOCKS modeling: run_experiment raises on any pinned-expectation
  mismatch before any fit; check_audit maps audit keys to manifest keys
  (regression tested); missing `_score` leak verification is itself a failure.
- The constant baseline is the TRAIN-fold UNCONDITIONAL mean floored at ε
  (previously, wrongly, E[Y | Y > 0]).
- Brier skill uses the TRAIN-fold prevalence as baseline, carried per test row
  from the same folds that produced the model's predictions (previously, wrongly,
  the pooled evaluation prevalence).
- Cluster bootstrap: O(n) per-vendor-totals fast path for MAE, verified exactly
  equivalent to explicit row resampling (same-seed test); missing vendor IDs
  form their own cluster via factorize.
- Parquet conversion: atomic (temp file + rename on success), tested across
  multiple chunks with nulls and `_score` preservation.
- RAR extraction: tool-specific argument construction per extractor; verified
  against the pinned archive (byte-identical products CSV).
- The H3 pre-split vendor-aggregate experiment is explicitly FUTURE WORK; the
  current contamination demo adds only `_score`.
- Results metadata now records python/sklearn/lightgbm versions and a data
  fingerprint (target + group columns hash).

## D10 — 2026-10-24: Phase 1 executed on the real data (commit 79831c7)
- `prepare-data` verified end-to-end: pinned archive SHA-256 match, 7-Zip
  extraction OK, chunked CSV→Parquet (2,411,358 rows, 5 row groups).
- The audit gate PASSED: every pinned expectation reproduced exactly (rows
  2,411,358; zero share 0.9681; 88,822 vendors; 3 missing vendor IDs; 0
  duplicate IDs; 105,540 total sales; 76,942 nonzero rows; `_score` leak
  fraction 1.0).
- Real-data defects found and fixed (each regression-tested):
  (a) Parquet loads as arrow-backed dtypes, which LightGBM rejects — feature
  builder now emits plain float64;
  (b) pandas `string`-dtype NA survives .astype(str) and factorize emits -1
  codes, crashing the cluster bootstrap — fillna("<missing>") before factorize;
  (c) group_kfold dropped NaN groups, which would have violated full coverage
  (3 rows with no vendor_id) — missing groups now form their own `<missing>`
  group; (d) fold masks vectorized (pd.Series.isin) for the 2.4M-row scale.
- Memory: ~1.6 GB RSS observed (process snapshot) during the grouped
  experiment; true peak not measured. Extraction verified by FULL-file
  SHA-256 (products CSV 38f598847d5a6d49506ee5da243577287741e8c68ac66c13660bc8ccb3599fad,
  identical to the reference extraction); Parquet SHA-256
  f0da5f47cf5b914b188957b9ddbe08f47b1bcd22cbcce6826db35ed61291627e.
- FIRST REAL RESULTS (single seed, grouped-vendor, leakage-safe): on Poisson
  deviance the ranking is hurdle 603,428 < Poisson-GBM 676,580 < train-mean
  baseline 766,960 — both models beat the prediction-time-valid constant
  baseline; hurdle stage-1: PR-AUC 0.1855 vs prevalence 0.0319, Brier skill
  +0.0531 (H4 direction, single seed). On MAE the median baselines win
  (0.0438 vs 0.1144/0.1018) — exactly the D8 caveat: MAE prefers the
  near-zero median under 96.8% zeros. Strata MAE grows with volume
  (0.070 / 0.884 / 3.285 / 13.218 across {0, 1-2, 3-10, >10}).
  These are SINGLE-SEED, SELLER-DISJOINT PILOT numbers with `_score` excluded:
  descriptive only, not formal hypothesis support (paired uncertainty analysis
  is Phase 4); no random-regime contrast, no tuning, feature set v0.
  Provenance: results.json records commit 08562cec, but the run included the
  Phase 1 fixes committed immediately after as 79831c7 — reproduce from
  79831c7 or later.
