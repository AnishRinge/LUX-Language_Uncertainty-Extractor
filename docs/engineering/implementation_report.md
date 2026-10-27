# Implementation record: 2026-10-27

## Delivered

- Stage A audit: audit_2026_10_27.md. Confirmed research limitations are separated
  from unresolved provenance/leakage concerns. Historical scripts remain unchanged.
- Prioritized plan and future evaluation protocol: implementation_plan.md.
- src/predictor/inference.py: strict pre-generation Python contract, explicit target
  and decoding binding, JSON artifact schema/integrity checks, cohort mean/median
  training, prompt-free structured log fields, bounded finite output, and policy.
- baseline_mean_v2.json / baseline_median_v2.json: constants fitted on the final
  cumulative strict snapshot, source hash, training-ID hash, predictor fingerprint,
  pipeline and format versions. export_baselines.py reproduces export without
  generation/evaluation and refuses overwrite. No executable model serialization.
- analyze_saved_oof.py and oof_analysis_v2.json: ID alignment, exact-once outer
  coverage, disjoint folds, training-fold baseline checks, fold MAE/RMSE/R², paired
  gains and relative gains against both baselines, seeded 2,000-resample bootstrap.
  v1 is an earlier engineering analysis before training-fold baseline validation;
  v2 is authoritative. Both are new engineering outputs, not historical benchmarks.
- tests/test_risk_inference.py and tests/test_saved_oof_analysis.py; README usage.

## Selection ruling

No learned predictor is promoted. Mean is the default cohort expected risk under
squared-error/expectation semantics; median is exposed for absolute-error decisions.
This is a deliberate statistical estimand choice, not selection on a favorable
single metric. A constant mean is not a calibrated individualized risk score, and
the zero median is not evidence of safety. Neither baseline can rank prompts.
Model loading or feature extraction failures are not replaced with zero vectors.
Hidden-state inference is explicitly unsupported by the v1 artifact pipeline.
The constant pipeline has an empty feature schema and no fitted preprocessing.

## Actual verification

- Initial new inference test collection failed as expected before implementation:
  ModuleNotFoundError for src.predictor.inference.
- Inference plus existing predictor tests: 47 passed in 200.95 seconds.
- Full tests directory at the first implementation snapshot: 355 passed in 245.33
  seconds. Later additions only strengthened OOF checks and added focused tests.
- Inference/OOF focused tests before review: 26 passed in 9.00 seconds.
- Final inference/OOF/export focused tests after the provenance fix: 27 passed in
  3.57 seconds. v2 export and end-to-end README-style smoke test succeeded: mean
  .20864197530864198, default action escalate, two provenance source hashes.
- Artifact export succeeded; saved-OOF v2 analysis succeeded for 39 configurations.
- Recomputed best learned MAE: svr|prompt .2141816293, versus median .2086419753.
  Mean MAE .3170642458, RMSE .3869038770, R² -.0006585193. Median RMSE
  .4394628221, R² -.2909933775. et|fusion RMSE .3834722152 and R² .0170135341
  improve these metrics, while MAE .3094088308 remains worse than median.
- Paired svr MAE gain against median: -.0055396539; conditional prompt bootstrap
  95% interval [-.0063548444, -.0047221604]. These intervals do not account for shared
  context dependence or search multiplicity and do not quantify retraining variance.
- git diff --check passed. Reviewed README diff and all new implementation files.
- Metadata inventory: 10,475 historical files under data/production and experiments;
  zero size/mtime changes since the post-audit inventory. This is a filesystem
  metadata preservation check, not a cryptographic attestation of every historical byte.
  No historical artifact was written by these changes.

## Resources and limits

A local synthetic microbenchmark with Python tracemalloc enabled measured artifact
load .00418 seconds and 1,000 baseline calls in .50757 seconds (mean .000508 seconds
per call), with 140,435 traced peak Python bytes. This excludes interpreter/import
startup, is not total process RAM, was measured while tests were running and is not
a production latency guarantee. No target model was loaded; hidden-state extraction
latency and RAM were not measured. Measuring them would require a separate bounded
single-prompt resource experiment before making deployment claims.

Individual uncertainty/calibration, OOD detection, cross-model support and validated
routing thresholds remain unsupported. Malformed prompt/config/artifact inputs raise
explicit errors. Missing predictions route to escalation. Uncalibrated threshold
routing requires caller opt-in; default policy always escalates. No service/API/UI
was added. Inference logs omit prompts; callers can serialize Prediction.to_dict()
with their own non-sensitive request identifiers and timestamps for monitoring.

## Remaining priorities

1. Recover the exact nested-CV runner and environment lock; reconcile all hidden
   rows, canonical features and evaluator records without regenerating outputs.
2. Harden old training joins/imputation in a separately tested compatibility change;
   audit shared context/topic duplication and stratified performance.
3. Evaluate training-fold-only calibration and context-grouped validation. Saved OOF
   predictions must not be reused as both calibrator training and unbiased testing.
4. Select policy thresholds on validation costs, then test prospective routing versus
   normal generation and budget-matched simple routing. The plan records outcomes
   and group analyses; no operational benefit is claimed from the current work.

No dependency installation, expensive generation/evaluation, commit or push occurred.
User changes present at startup were preserved.

## Independent review and follow-up

A fresh reviewer ran the 26 focused tests and found an important exporter provenance
gap: row counts and risk checks alone did not bind a replacement input snapshot to
the frozen model/config. The exporter now pins the audited target and generation
file SHA256 hashes and embeds both in artifact provenance. New snapshots fail closed
pending a provenance review. It also validates all count types, excluded outcomes
and the strict eligible count. v2 artifacts supersede the preserved v1 engineering
artifacts; predictions are unchanged. A regression test first failed to import the
new validation helper before implementation. Current README selects v2. The reviewer
confirmed the issue was resolved and found no new important issues in the delta.
Final historical metadata recheck remained at zero changes; git diff --check passed.
