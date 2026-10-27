# Milestone 2 analysis contract (declared before computing routing results)

Use the saved five outer folds and exactly the 486 strict eligible prompt IDs.
Do not fit estimators, select models or tune thresholds. All risk-bound-valid saved
configurations are reported; configurations with any nonfinite/out-of-bounds risk
are excluded without clipping or dropping individual prompts. Deduplicate baseline
feature-family copies only after checking identical predictions.

Threshold sensitivity grid: 0, .05, .1, .2, .3, .5, .75, 1. A score >= threshold
routes to verification; otherwise normal generation. This is a descriptive grid,
not validated operational thresholds. No-risk-estimation reference allows all.
Outcome-independent SHA256 ID routing uses budgets 0, .1, .25, .5, .75, 1, rounding
down per outer test fold. Candidate comparisons also use exactly count-matched hash
routing per outer test fold. Hash salt: lux-routing-20261027. Ties are resolved by ID.

Report normal-path mean empirical risk, normal-path fraction with any UNRELIABLE
attempt, escalation fraction, captured unreliable mass, and workload-normalized
remaining risk under explicit residual-risk multipliers 0, .5, 1 on escalated prompts.
No actual verification benefit is measured. Aggregate and all five fold results.

Paired uncertainty: seed 20261027; 2,000 prompt bootstrap replicates stratified by
saved outer test fold; same resampled IDs for candidate and comparator. Ratios with
zero denominators are undefined, never zero. Report support for bootstrap intervals.
Intervals condition on fitted scores and masks; they omit retraining, shared-context
dependence and multiple-comparison correction. No significance-based winner.

Leakage guard: adapting a model/threshold requires a separate validation cohort for
each outer model with lineage showing all score-producing fits exclude the outer
test set. Other folds' ordinary OOF predictions are not sufficient. Missing evidence
must raise, not permit threshold optimization. No threshold selection runs here.

Implementation: additive analysis.py, test_analysis.py, results.json and report.md
in this directory. Validate strict target counts, ID joins, fold coverage, duplicate
IDs/configurations, baseline train-only constants and artifact hashes. Tests first,
then focused existing tests; execute analysis read-only; verify historical inventory.
New output files use exclusive creation. No changes to production inference/policy.
