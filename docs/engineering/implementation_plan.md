# Baseline inference implementation plan

Goal: a modest usable contract without promoting unsupported learned performance.
Architecture: JSON-only constant baseline artifacts; no pickle execution, target-model
loading, network calls, or hidden-state fallback. Python standard library plus existing
NumPy for offline fitting/evaluation. User brief is the governing specification.

1. Add tests in tests/test_risk_inference.py for strict prompt/config/artifact schemas,
   reproducible artifact round trips, leakage rejection and conservative routing.
2. Add src/predictor/inference.py: versioned mean/median training artifacts, provenance
   hashes, configuration binding, structured results, prompt-free logging, explicit
   unsupported uncertainty and out-of-domain warnings. Use mean by default as cohort
   expected risk; expose median explicitly for absolute-error decisions. Neither is
   evidence of individual predictive skill. No learned predictor is promoted.
3. Add offline saved-OOF analysis with ID joins, identical fold checks, all three
   metrics per fold, paired effects versus both baselines, seeded bootstrap intervals.
   Intervals describe aggregate sampling variability, not individual certainty.
4. Export new artifacts from the final strict snapshot only, recording source hashes;
   fit constants without claiming a new held-out evaluation. Preserve historical data.
5. Document setup, policy threshold selection, future calibration/generalization and
   downstream evaluation protocol. Run focused and broader tests; inspect diff and
   verify historical file fingerprints remain unchanged.

Recovery: all implementation modules and engineering artifacts are additive; README
gets an additive current-status section. No historical runner or experiment is edited.
No commits/pushes, dependency installation, generation or evaluator requests.

Execution record: steps 1–5 completed for the baseline-only scope; see
implementation_report.md for actual tests, limitations, review and provenance fix.
Full learned-model promotion, calibration and external/operational experiments are
future work and were not executed.

Calibration next step: fit a proposed calibrator using training-only cross-fitted
predictions inside each outer fold, select it by a prespecified criterion, and compare
held-out error plus calibration curves with bin support/uncertainty. Current OOF results
are evaluation data; fitting and evaluating a calibrator on them would be optimistic.

Policy next step: choose thresholds on validation data using costs of unreliable
answers, verification, abstention, latency and compute; freeze before prospective test.
Compare normal generation, LUX routing and a constant/random routing comparator at
matched verification budgets. Measure unreliable-answer rate, low-risk unreliable
outcomes (false reassurance), coverage, abstention, verification rate, latency, cost and
group performance. Stratify by answerability, question type and length where supported;
use context/topic groups to assess shared-context contamination. Optional independent
label validation can quantify evaluator bias. No downstream benefits are measured here.
