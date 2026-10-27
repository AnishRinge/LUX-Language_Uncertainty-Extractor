# LUX Milestone 2: retrospective routing evaluation

## Verdict and scope

**PARTIAL PASS: descriptive analysis executed; a leakage-safe selected policy cannot
be certified from the available artifacts.** No model or routing threshold was
selected. No model training, answer generation or evaluator call occurred. Results
are retrospective sensitivity simulations, not measured improvements in a live system.

The analysis includes every risk-bound-valid saved configuration at every declared
threshold, with the same 486 eligible prompts and original five outer folds. Saved
OOF predictions exclude each prompt from its outer estimator's training set according
to the saved assignment, but the original nested-CV runner and complete inner
preprocessing/model-selection lineage are unavailable. ID/target alignment alone
cannot prove those training boundaries. Learned results remain provisional.

## Sources and verified contract

The frozen target is Qwen/Qwen3-1.7B, revision
70d244cc86ccca08cf5af4e1e306ecf908b1ad5e, CPU, thinking disabled,
do_sample=True, temperature=.7, top_p=.8, top_k=20, max_new_tokens=512.
Source hashes are checked against the previously audited baseline provenance for
the strict target snapshot and generation artifact. The deployment constant fitted
on all targets is NOT used to evaluate baseline routing: saved training-fold-only
mean and median OOF constants are used instead.

Eligible outcomes are UNRELIABLE/10 from ten SUCCESS binary RELIABLE/UNRELIABLE
judgments, with inadequate, missing, failed and manual-review outcomes excluded.
Counts, unique IDs/dataset indices, risks, saved y values and every OOF target are
checked. These are evaluator-derived proxy judgments, not independently verified
truth or next-answer probabilities. Excluding incomplete prompts can induce bias.

N=486; zero-risk prompts=367 (75.51%); mean risk=0.2086419753. The eligible cohort contains 1014 UNRELIABLE judgments out of 4,860 attempts.

| Source | Exact path | SHA256 |
|---|---|---|
| targets | data/production/cp4_7_0_gemini_3_5_flash_lite/targets_batch_05.json | 9f4a288540d41b31140248a690ad2759cff2d4cc66145fd0643ed4c28cf90914 |
| generations | data/production/squad_v2_expanded_500/generations.json | 1a7babd61f817642d718c96e98b14d40196c0c60e99fbfc5dea463ccea16f3b8 |
| y | experiments/final_cohort_2026_10_24/y.npy | cbc44a48374b370f49c388a36ae60737cf20b8287f934e6f74077f4d4e6b83a5 |
| ids | experiments/final_cohort_2026_10_24/pids.npy | 9613e59937e911f7fc97275d1bfda3e2155b0b90f720b86da8caa3aeb5abb531 |
| folds | experiments/expanded_model_selection_2026_10_24/fold_assignments.json | 49ca50c6a6d1c5a0cfffa155234a642b7078173a989051061d71eac2c8ca5f14 |
| oof | experiments/expanded_model_selection_2026_10_24/nested_cv_results.json | 139772c8cf2ba821c2cc97add1bad96da5c08c337f41e38465df2f45349b5632 |
| provenance | docs/engineering/baseline_mean_v2.json | 8fad366f70f2ee1b399661487bb0753ddb0a9103dde0da2c33fc3899374e2ca1 |

Analysis SHA256: c5670c4a5f674eab2e6552f9d37d8f50a72e55aa12746891bd112faf61dd0c8d. Declared contract SHA256: dfb1252cbe9f4a10d2db8a583cca2aa1ed47cab03aed535c875e6419fadf1140.

The hidden representation metadata describes mean over the layer axis of saved
(486,29,2048) per-layer token-mean arrays. This differs from the canonical last-layer
token-mean helper. No new hidden extraction or reinterpretation was performed.

## Leakage-safe scope and missing selection evidence

Threshold grid 0, .05, .1, .2, .3, .5, .75, 1 was written in plan.md before routing
results were computed. Every threshold/configuration is retained. No outcome-based
choice, cost fitting, winning-model ranking or threshold optimizer runs. These are
descriptive curves; the grid is not scientifically validated and is not installed
as a deployment policy. Prior access to historical research outcomes means this is
not a preregistered independent confirmation experiment.

Ordinary pooled OOF predictions cannot be used naively for nested threshold/model
selection: when evaluating outer fold A, OOF scores in fold B may come from a fit
whose training data included A's outcomes. Using those rows to tune A's policy can
carry information from A back into selection. The saved files do not contain the
outer-A-specific inner-validation predictions and lineage needed to exclude this.
require_selection_evidence rejects absent/contaminated lineage; it is a guard, not
an optimizer or proof that unprovided lineage exists.

The minimum next step is to recover original outer-model-specific validation score
artifacts and fit/preprocessing/selection IDs for each saved outer fold. If unavailable,
an explicitly authorized predictor-only nested evaluation is needed: keep these exact
outer folds; generate inner validation predictions using only each outer training
partition; select model and thresholds there using declared costs; freeze the policy;
then score the untouched outer test partition once. Do not use that partition for
selection. No new target-model generations or evaluator labels are required for that
retrospective protocol. Preserve all existing outputs. For an independent confirmatory
generalization claim, additionally use a genuinely untouched cohort and fixed pipeline.

## Policy interfaces and references

Existing production API remains unchanged: predict_risk(..., artifact=...) returns
a structured estimate; Policy(verify_at, escalate_at, allow_uncalibrated=False)
supports allow/verify/escalate. Its default escalates and uncalibrated routing needs
explicit opt-in. It is NOT the no-risk-estimation comparator.

The research interface ThresholdRule(threshold, high_action='verify', low_action='allow')
requires an explicit finite threshold in [0,1]. score >= threshold uses high_action
(verify or escalate); all other scores use allow. It rejects invalid scores without
clipping/imputation. It accepts scores only, never targets or generation fields.
This binary sensitivity rule collapses the production policy's non-allow actions;
for verify_at < escalate_at, normal-path coverage depends only on verify_at. Actual
verification versus escalation effects have not been measured separately.

References: (A) no-risk-estimation normal generation allows all prompts; (B,C) saved
median/mean baselines; (D) all valid learned OOF configurations without new model
selection; (E) deterministic non-predictive SHA256 prompt-ID routing. Fixed salt is
lux-routing-20261027. Hash budgets 0,.1,.25,.5,.75,1 are rounded down in each saved
outer test fold. Additional candidate-specific hash references match its exact number
of escalated prompts in every fold; only IDs and counts enter that mask, not outcomes.
One fixed hash realization is used, not a random-routing population estimate.

Four duplicate mean/median hidden/fusion copies are collapsed after exact equality
checks. A configuration with any out-of-bounds/nonfinite risk is wholly excluded;
no prompt is selectively removed and no score clipped. This is contract eligibility,
not selection based on target performance. Scores in [0,1] are not thereby calibrated.

## Metric definitions and hypothetical assumptions

Let y_i=UNRELIABLE_i/10, E be escalated and N be normal-generation prompts.

- Normal-path empirical unreliable rate and false reassurance generation proxy:
  sum(y_i in N)/|N|. This is an average of ten-attempt prompt risks, not a newly
  observed answer failure rate. The two names refer to the same quantity here.
- Additional prompt-level false reassurance descriptor: fraction of N with y_i>0
  (at least one unreliable attempt). Do not confuse it with the generation proxy.
- Escalation rate: |E|/486; coverage: |N|/486. Captured unreliable mass:
  sum(y_i in E)/sum(y_i). Equal ten-attempt denominators make it also the fraction
  of eligible UNRELIABLE judgments routed to escalation.
- Hypothetical remaining risk per original prompt:
  [sum(y_i in N)+q*sum(y_i in E)]/486 for residual multipliers q=0,.5,1.
  q=0 assumes perfect removal; q=.5 assumes halved empirical risk; q=1 assumes no
  benefit. These assumptions are NOT measured verification effectiveness. With
  abstention, report coverage and normal-path risk instead of treating abstentions
  as reliable answers. Zero denominators yield null/undefined, never reassuring zero.

Cost interface Costs(unreliable, verify, escalate) and
expected_cost(outcomes, actions, costs, residual_risk_fraction) computes an offline
scenario: mean(C_unreliable*y_remaining + C_verify*I_verify + C_escalate*I_escalate).
Costs must be explicit nonnegative finite values; no cost vector is selected or used
to declare a winner here. A team should set error severity, review cost, latency and
compute cost in comparable units; use inner validation to select thresholds under
those costs and coverage/budget constraints; freeze them before outer testing.
No operational cost/latency measurements for learned routing are available.

## Uncertainty method

2,000 paired prompt-bootstrap replicates, seed 20261027, sampled within original
outer test folds. Candidate and reference use identical sampled IDs. results.json
contains point differences (candidate minus reference), percentile 95% intervals,
and number of defined replicates for every metric versus normal generation and
count-matched hash routing. Negative normal-risk difference favors the candidate;
positive capture difference favors it; lower escalation saves work. Undefined
ratios remain null. All five per-fold performance records are saved.

Intervals condition on fixed predictions, thresholds, hash masks and fold membership.
They do not include retraining/selection variance, dependence from shared contexts,
or multiple-comparison correction. Bootstrap does not repair missing provenance or
selection leakage. No significance-based model/threshold recommendation is made.

## Complete normal-generation and non-predictive references

| Reference | Escalated | Normal proxy risk | Captured unreliable mass | Any-unreliable prompt fraction in normal path |
|---|---:|---:|---:|---:|
| normal_generation | 0.00% | 20.86% | 0.00% | 24.49% |
| hash_budget_0.0 | 0.00% | 20.86% | 0.00% | 24.49% |
| hash_budget_0.1 | 9.26% | 19.59% | 14.79% | 23.58% |
| hash_budget_0.25 | 24.69% | 20.41% | 26.33% | 24.59% |
| hash_budget_0.5 | 49.59% | 22.08% | 46.65% | 27.35% |
| hash_budget_0.75 | 74.28% | 22.64% | 72.09% | 28.80% |
| hash_budget_1.0 | 100.00% | undefined | 100.00% | undefined |

## Complete threshold trade-offs (no selected policy)

All valid configurations and all declared thresholds follow. q=0 remaining is an
idealized per-original-prompt scenario, not an observed improvement. Full q=.5/q=1
results, per-fold distributions and paired intervals are in results.json.

| Configuration | Threshold | Escalated | Normal proxy risk / false reassurance | Captured mass | Any-unreliable normal prompts | q=0 remaining |
|---|---:|---:|---:|---:|---:|---:|
| dummy_mean|prompt | 0 | 100.00% | undefined | 100.00% | undefined | 0.00% |
| dummy_mean|prompt | 0.05 | 100.00% | undefined | 100.00% | undefined | 0.00% |
| dummy_mean|prompt | 0.1 | 100.00% | undefined | 100.00% | undefined | 0.00% |
| dummy_mean|prompt | 0.2 | 100.00% | undefined | 100.00% | undefined | 0.00% |
| dummy_mean|prompt | 0.3 | 0.00% | 20.86% | 0.00% | 24.49% | 20.86% |
| dummy_mean|prompt | 0.5 | 0.00% | 20.86% | 0.00% | 24.49% | 20.86% |
| dummy_mean|prompt | 0.75 | 0.00% | 20.86% | 0.00% | 24.49% | 20.86% |
| dummy_mean|prompt | 1 | 0.00% | 20.86% | 0.00% | 24.49% | 20.86% |
| dummy_median|prompt | 0 | 100.00% | undefined | 100.00% | undefined | 0.00% |
| dummy_median|prompt | 0.05 | 0.00% | 20.86% | 0.00% | 24.49% | 20.86% |
| dummy_median|prompt | 0.1 | 0.00% | 20.86% | 0.00% | 24.49% | 20.86% |
| dummy_median|prompt | 0.2 | 0.00% | 20.86% | 0.00% | 24.49% | 20.86% |
| dummy_median|prompt | 0.3 | 0.00% | 20.86% | 0.00% | 24.49% | 20.86% |
| dummy_median|prompt | 0.5 | 0.00% | 20.86% | 0.00% | 24.49% | 20.86% |
| dummy_median|prompt | 0.75 | 0.00% | 20.86% | 0.00% | 24.49% | 20.86% |
| dummy_median|prompt | 1 | 0.00% | 20.86% | 0.00% | 24.49% | 20.86% |
| svr|prompt | 0 | 100.00% | undefined | 100.00% | undefined | 0.00% |
| svr|prompt | 0.05 | 0.00% | 20.86% | 0.00% | 24.49% | 20.86% |
| svr|prompt | 0.1 | 0.00% | 20.86% | 0.00% | 24.49% | 20.86% |
| svr|prompt | 0.2 | 0.00% | 20.86% | 0.00% | 24.49% | 20.86% |
| svr|prompt | 0.3 | 0.00% | 20.86% | 0.00% | 24.49% | 20.86% |
| svr|prompt | 0.5 | 0.00% | 20.86% | 0.00% | 24.49% | 20.86% |
| svr|prompt | 0.75 | 0.00% | 20.86% | 0.00% | 24.49% | 20.86% |
| svr|prompt | 1 | 0.00% | 20.86% | 0.00% | 24.49% | 20.86% |
| knn|prompt | 0 | 100.00% | undefined | 100.00% | undefined | 0.00% |
| knn|prompt | 0.05 | 60.08% | 20.52% | 60.75% | 24.23% | 8.19% |
| knn|prompt | 0.1 | 57.00% | 19.38% | 60.06% | 22.97% | 8.33% |
| knn|prompt | 0.2 | 46.09% | 20.04% | 48.22% | 23.28% | 10.80% |
| knn|prompt | 0.3 | 31.69% | 19.91% | 34.81% | 22.89% | 13.60% |
| knn|prompt | 0.5 | 10.29% | 20.50% | 11.83% | 23.39% | 18.40% |
| knn|prompt | 0.75 | 0.62% | 20.64% | 1.68% | 24.22% | 20.51% |
| knn|prompt | 1 | 0.00% | 20.86% | 0.00% | 24.49% | 20.86% |
| dt|prompt | 0 | 100.00% | undefined | 100.00% | undefined | 0.00% |
| dt|prompt | 0.05 | 75.31% | 22.33% | 73.57% | 25.83% | 5.51% |
| dt|prompt | 0.1 | 68.31% | 22.08% | 66.47% | 25.32% | 7.00% |
| dt|prompt | 0.2 | 60.08% | 22.37% | 57.20% | 25.77% | 8.93% |
| dt|prompt | 0.3 | 13.79% | 21.58% | 10.85% | 25.54% | 18.60% |
| dt|prompt | 0.5 | 9.26% | 21.72% | 5.52% | 25.62% | 19.71% |
| dt|prompt | 0.75 | 4.94% | 21.15% | 3.65% | 24.89% | 20.10% |
| dt|prompt | 1 | 1.65% | 21.00% | 0.99% | 24.69% | 20.66% |
| rf|prompt | 0 | 100.00% | undefined | 100.00% | undefined | 0.00% |
| rf|prompt | 0.05 | 99.18% | 25.00% | 99.01% | 25.00% | 0.21% |
| rf|prompt | 0.1 | 91.15% | 31.16% | 86.79% | 34.88% | 2.76% |
| rf|prompt | 0.2 | 48.15% | 22.30% | 44.58% | 25.40% | 11.56% |
| rf|prompt | 0.3 | 15.02% | 21.53% | 12.33% | 25.18% | 18.29% |
| rf|prompt | 0.5 | 1.65% | 21.00% | 0.99% | 24.69% | 20.66% |
| rf|prompt | 0.75 | 0.21% | 20.70% | 0.99% | 24.33% | 20.66% |
| rf|prompt | 1 | 0.00% | 20.86% | 0.00% | 24.49% | 20.86% |
| et|prompt | 0 | 100.00% | undefined | 100.00% | undefined | 0.00% |
| et|prompt | 0.05 | 99.38% | 40.00% | 98.82% | 66.67% | 0.25% |
| et|prompt | 0.1 | 95.06% | 17.92% | 95.76% | 25.00% | 0.88% |
| et|prompt | 0.2 | 52.26% | 20.04% | 54.14% | 23.28% | 9.57% |
| et|prompt | 0.3 | 12.14% | 21.03% | 11.44% | 24.59% | 18.48% |
| et|prompt | 0.5 | 0.41% | 20.95% | 0.00% | 24.59% | 20.86% |
| et|prompt | 0.75 | 0.21% | 20.91% | 0.00% | 24.54% | 20.86% |
| et|prompt | 1 | 0.00% | 20.86% | 0.00% | 24.49% | 20.86% |
| hgb|prompt | 0 | 100.00% | undefined | 100.00% | undefined | 0.00% |
| hgb|prompt | 0.05 | 99.59% | 90.00% | 98.22% | 100.00% | 0.37% |
| hgb|prompt | 0.1 | 93.00% | 16.18% | 94.58% | 20.59% | 1.13% |
| hgb|prompt | 0.2 | 51.85% | 21.28% | 50.89% | 24.36% | 10.25% |
| hgb|prompt | 0.3 | 16.87% | 21.91% | 12.72% | 25.99% | 18.21% |
| hgb|prompt | 0.5 | 0.00% | 20.86% | 0.00% | 24.49% | 20.86% |
| hgb|prompt | 0.75 | 0.00% | 20.86% | 0.00% | 24.49% | 20.86% |
| hgb|prompt | 1 | 0.00% | 20.86% | 0.00% | 24.49% | 20.86% |
| lasso|hidden | 0 | 100.00% | undefined | 100.00% | undefined | 0.00% |
| lasso|hidden | 0.05 | 100.00% | undefined | 100.00% | undefined | 0.00% |
| lasso|hidden | 0.1 | 100.00% | undefined | 100.00% | undefined | 0.00% |
| lasso|hidden | 0.2 | 100.00% | undefined | 100.00% | undefined | 0.00% |
| lasso|hidden | 0.3 | 0.00% | 20.86% | 0.00% | 24.49% | 20.86% |
| lasso|hidden | 0.5 | 0.00% | 20.86% | 0.00% | 24.49% | 20.86% |
| lasso|hidden | 0.75 | 0.00% | 20.86% | 0.00% | 24.49% | 20.86% |
| lasso|hidden | 1 | 0.00% | 20.86% | 0.00% | 24.49% | 20.86% |
| elasticnet|hidden | 0 | 100.00% | undefined | 100.00% | undefined | 0.00% |
| elasticnet|hidden | 0.05 | 100.00% | undefined | 100.00% | undefined | 0.00% |
| elasticnet|hidden | 0.1 | 100.00% | undefined | 100.00% | undefined | 0.00% |
| elasticnet|hidden | 0.2 | 79.22% | 20.50% | 79.59% | 23.76% | 4.26% |
| elasticnet|hidden | 0.3 | 0.21% | 20.91% | 0.00% | 24.54% | 20.86% |
| elasticnet|hidden | 0.5 | 0.00% | 20.86% | 0.00% | 24.49% | 20.86% |
| elasticnet|hidden | 0.75 | 0.00% | 20.86% | 0.00% | 24.49% | 20.86% |
| elasticnet|hidden | 1 | 0.00% | 20.86% | 0.00% | 24.49% | 20.86% |
| svr|hidden | 0 | 100.00% | undefined | 100.00% | undefined | 0.00% |
| svr|hidden | 0.05 | 13.17% | 17.80% | 25.94% | 21.09% | 15.45% |
| svr|hidden | 0.1 | 0.62% | 20.81% | 0.89% | 24.22% | 20.68% |
| svr|hidden | 0.2 | 0.00% | 20.86% | 0.00% | 24.49% | 20.86% |
| svr|hidden | 0.3 | 0.00% | 20.86% | 0.00% | 24.49% | 20.86% |
| svr|hidden | 0.5 | 0.00% | 20.86% | 0.00% | 24.49% | 20.86% |
| svr|hidden | 0.75 | 0.00% | 20.86% | 0.00% | 24.49% | 20.86% |
| svr|hidden | 1 | 0.00% | 20.86% | 0.00% | 24.49% | 20.86% |
| knn|hidden | 0 | 100.00% | undefined | 100.00% | undefined | 0.00% |
| knn|hidden | 0.05 | 67.90% | 15.38% | 76.33% | 17.95% | 4.94% |
| knn|hidden | 0.1 | 65.43% | 17.80% | 70.51% | 20.24% | 6.15% |
| knn|hidden | 0.2 | 44.44% | 17.96% | 52.17% | 21.85% | 9.98% |
| knn|hidden | 0.3 | 31.48% | 19.25% | 36.79% | 22.82% | 13.19% |
| knn|hidden | 0.5 | 8.23% | 21.01% | 7.59% | 24.44% | 19.28% |
| knn|hidden | 0.75 | 1.03% | 20.89% | 0.89% | 24.32% | 20.68% |
| knn|hidden | 1 | 0.21% | 20.74% | 0.79% | 24.33% | 20.70% |
| dt|hidden | 0 | 100.00% | undefined | 100.00% | undefined | 0.00% |
| dt|hidden | 0.05 | 67.49% | 24.18% | 62.33% | 26.58% | 7.86% |
| dt|hidden | 0.1 | 44.65% | 22.42% | 40.53% | 26.02% | 12.41% |
| dt|hidden | 0.2 | 20.16% | 19.41% | 25.74% | 22.68% | 15.49% |
| dt|hidden | 0.3 | 19.75% | 19.82% | 23.77% | 23.08% | 15.91% |
| dt|hidden | 0.5 | 16.87% | 19.63% | 21.79% | 22.77% | 16.32% |
| dt|hidden | 0.75 | 11.93% | 19.95% | 15.78% | 23.36% | 17.57% |
| dt|hidden | 1 | 3.70% | 20.43% | 5.72% | 23.93% | 19.67% |
| rf|hidden | 0 | 100.00% | undefined | 100.00% | undefined | 0.00% |
| rf|hidden | 0.05 | 100.00% | undefined | 100.00% | undefined | 0.00% |
| rf|hidden | 0.1 | 95.06% | 30.83% | 92.70% | 37.50% | 1.52% |
| rf|hidden | 0.2 | 40.95% | 17.25% | 51.18% | 20.56% | 10.19% |
| rf|hidden | 0.3 | 11.32% | 17.98% | 23.57% | 21.11% | 15.95% |
| rf|hidden | 0.5 | 0.21% | 20.91% | 0.00% | 24.54% | 20.86% |
| rf|hidden | 0.75 | 0.00% | 20.86% | 0.00% | 24.49% | 20.86% |
| rf|hidden | 1 | 0.00% | 20.86% | 0.00% | 24.49% | 20.86% |
| et|hidden | 0 | 100.00% | undefined | 100.00% | undefined | 0.00% |
| et|hidden | 0.05 | 98.35% | 5.00% | 99.61% | 12.50% | 0.08% |
| et|hidden | 0.1 | 93.83% | 19.00% | 94.38% | 23.33% | 1.17% |
| et|hidden | 0.2 | 48.97% | 17.94% | 56.11% | 20.97% | 9.16% |
| et|hidden | 0.3 | 15.84% | 18.36% | 25.94% | 21.76% | 15.45% |
| et|hidden | 0.5 | 1.03% | 20.50% | 2.76% | 24.12% | 20.29% |
| et|hidden | 0.75 | 0.00% | 20.86% | 0.00% | 24.49% | 20.86% |
| et|hidden | 1 | 0.00% | 20.86% | 0.00% | 24.49% | 20.86% |
| lasso|fusion | 0 | 100.00% | undefined | 100.00% | undefined | 0.00% |
| lasso|fusion | 0.05 | 100.00% | undefined | 100.00% | undefined | 0.00% |
| lasso|fusion | 0.1 | 100.00% | undefined | 100.00% | undefined | 0.00% |
| lasso|fusion | 0.2 | 100.00% | undefined | 100.00% | undefined | 0.00% |
| lasso|fusion | 0.3 | 0.00% | 20.86% | 0.00% | 24.49% | 20.86% |
| lasso|fusion | 0.5 | 0.00% | 20.86% | 0.00% | 24.49% | 20.86% |
| lasso|fusion | 0.75 | 0.00% | 20.86% | 0.00% | 24.49% | 20.86% |
| lasso|fusion | 1 | 0.00% | 20.86% | 0.00% | 24.49% | 20.86% |
| elasticnet|fusion | 0 | 100.00% | undefined | 100.00% | undefined | 0.00% |
| elasticnet|fusion | 0.05 | 100.00% | undefined | 100.00% | undefined | 0.00% |
| elasticnet|fusion | 0.1 | 100.00% | undefined | 100.00% | undefined | 0.00% |
| elasticnet|fusion | 0.2 | 79.22% | 20.50% | 79.59% | 23.76% | 4.26% |
| elasticnet|fusion | 0.3 | 0.21% | 20.91% | 0.00% | 24.54% | 20.86% |
| elasticnet|fusion | 0.5 | 0.00% | 20.86% | 0.00% | 24.49% | 20.86% |
| elasticnet|fusion | 0.75 | 0.00% | 20.86% | 0.00% | 24.49% | 20.86% |
| elasticnet|fusion | 1 | 0.00% | 20.86% | 0.00% | 24.49% | 20.86% |
| svr|fusion | 0 | 100.00% | undefined | 100.00% | undefined | 0.00% |
| svr|fusion | 0.05 | 13.17% | 17.80% | 25.94% | 21.09% | 15.45% |
| svr|fusion | 0.1 | 0.62% | 20.81% | 0.89% | 24.22% | 20.68% |
| svr|fusion | 0.2 | 0.00% | 20.86% | 0.00% | 24.49% | 20.86% |
| svr|fusion | 0.3 | 0.00% | 20.86% | 0.00% | 24.49% | 20.86% |
| svr|fusion | 0.5 | 0.00% | 20.86% | 0.00% | 24.49% | 20.86% |
| svr|fusion | 0.75 | 0.00% | 20.86% | 0.00% | 24.49% | 20.86% |
| svr|fusion | 1 | 0.00% | 20.86% | 0.00% | 24.49% | 20.86% |
| knn|fusion | 0 | 100.00% | undefined | 100.00% | undefined | 0.00% |
| knn|fusion | 0.05 | 68.93% | 16.36% | 75.64% | 18.54% | 5.08% |
| knn|fusion | 0.1 | 66.46% | 17.55% | 71.79% | 19.63% | 5.88% |
| knn|fusion | 0.2 | 45.88% | 17.49% | 54.64% | 20.91% | 9.47% |
| knn|fusion | 0.3 | 29.63% | 18.16% | 38.76% | 21.35% | 12.78% |
| knn|fusion | 0.5 | 8.23% | 20.58% | 9.47% | 23.99% | 18.89% |
| knn|fusion | 0.75 | 1.03% | 20.89% | 0.89% | 24.32% | 20.68% |
| knn|fusion | 1 | 0.21% | 20.74% | 0.79% | 24.33% | 20.70% |
| dt|fusion | 0 | 100.00% | undefined | 100.00% | undefined | 0.00% |
| dt|fusion | 0.05 | 67.70% | 20.64% | 68.05% | 24.84% | 6.67% |
| dt|fusion | 0.1 | 29.63% | 19.80% | 33.23% | 23.10% | 13.93% |
| dt|fusion | 0.2 | 21.81% | 18.76% | 29.68% | 21.84% | 14.67% |
| dt|fusion | 0.3 | 15.23% | 19.15% | 22.19% | 22.33% | 16.23% |
| dt|fusion | 0.5 | 15.23% | 19.15% | 22.19% | 22.33% | 16.23% |
| dt|fusion | 0.75 | 12.76% | 19.01% | 20.51% | 22.17% | 16.58% |
| dt|fusion | 1 | 6.79% | 20.15% | 9.96% | 23.40% | 18.79% |
| rf|fusion | 0 | 100.00% | undefined | 100.00% | undefined | 0.00% |
| rf|fusion | 0.05 | 100.00% | undefined | 100.00% | undefined | 0.00% |
| rf|fusion | 0.1 | 95.68% | 22.86% | 95.27% | 23.81% | 0.99% |
| rf|fusion | 0.2 | 40.95% | 16.97% | 51.97% | 20.21% | 10.02% |
| rf|fusion | 0.3 | 12.35% | 18.73% | 21.30% | 21.83% | 16.42% |
| rf|fusion | 0.5 | 0.21% | 20.91% | 0.00% | 24.54% | 20.86% |
| rf|fusion | 0.75 | 0.00% | 20.86% | 0.00% | 24.49% | 20.86% |
| rf|fusion | 1 | 0.00% | 20.86% | 0.00% | 24.49% | 20.86% |
| et|fusion | 0 | 100.00% | undefined | 100.00% | undefined | 0.00% |
| et|fusion | 0.05 | 99.59% | 50.00% | 99.01% | 50.00% | 0.21% |
| et|fusion | 0.1 | 95.88% | 6.50% | 98.72% | 10.00% | 0.27% |
| et|fusion | 0.2 | 45.88% | 16.96% | 56.02% | 19.77% | 9.18% |
| et|fusion | 0.3 | 16.05% | 17.70% | 28.80% | 20.83% | 14.86% |
| et|fusion | 0.5 | 1.03% | 20.64% | 2.07% | 24.12% | 20.43% |
| et|fusion | 0.75 | 0.21% | 20.89% | 0.10% | 24.33% | 20.84% |
| et|fusion | 1 | 0.00% | 20.86% | 0.00% | 24.49% | 20.86% |

At any strictly positive threshold the zero median allows every prompt; at zero it
escalates all. The training-fold means are each constant within their test fold and
cannot rank prompts within a fold. A low regression MAE therefore does not establish
routing utility. Learned curves may show retrospective concentration of risk, but
the complete tables and conditional paired intervals do not justify selecting them.

## Contract-excluded configurations

| Configuration | Invalid risk predictions (of 486) |
|---|---:|
| ridge|prompt | 1 |
| lasso|prompt | 1 |
| elasticnet|prompt | 1 |
| gbr|prompt | 3 |
| mlp|prompt | 5 |
| ridge|hidden | 135 |
| gbr|hidden | 3 |
| hgb|hidden | 4 |
| mlp|hidden | 279 |
| ridge|fusion | 130 |
| gbr|fusion | 2 |
| hgb|fusion | 9 |
| mlp|fusion | 321 |

## Execution, tests, review and preservation

Activation in every execution shell used process-only policy bypass, followed by
. .\.venv\Scripts\Activate.ps1; sys.executable was verified against the project
.venv/Scripts/python.exe. No machine execution policy was changed.

Commands from the repository root:

```powershell
python -m pytest experiments/routing_policy_evaluation_2026_10_27/test_analysis.py -q --tb=short
python -m pytest experiments/routing_policy_evaluation_2026_10_27/test_analysis.py tests/test_saved_oof_analysis.py tests/test_risk_inference.py tests/test_inference_end_to_end.py tests/test_baseline_export.py -q --tb=short
python experiments/routing_policy_evaluation_2026_10_27/analysis.py
python experiments/routing_policy_evaluation_2026_10_27/write_report.py
git diff --check
```

Initial tests failed before analysis.py existed (FileNotFoundError). First implemented
tests: 13 passed in .82 s; expanded analysis plus relevant existing suite: 90 passed
in 11.61 s. Final focused/relevant run: 91 passed in 9.18 s, including 16 new routing
tests. Tests cover metric denominators, hypothetical assumptions, costs, ID order,
duplicate/incorrect targets, strict eligibility, provenance mismatch, fold overlap,
exact coverage, five-fold contract, threshold boundaries, invalid risk rejection,
outcome-free deterministic routing, hash budgets and paired bootstrap behavior.

Read-only independent review found no important scientific/correctness issues. Its
minor recommendation to require exactly five source folds was implemented and tested.
The final analysis completed; outputs refuse overwrite. results_initial.json preserves
the first completed calculation; results.json adds method/provenance hashes and the
explicit five-fold guard, with unchanged numerical routing results. To reproduce,
use a fresh workspace copy with the new output files absent; compare JSON numerically
and verify the recorded source/code hashes. Do not remove historical research files.

All changes are confined to this new directory. Before/after data/experiments inventory
excludes it. Historical artifact preservation and final diff results are recorded
in preservation_check.json. No staging, commit or push. No model fitting, model search,
generation or evaluator pipeline was invoked.

No actual verification/escalation outcomes are available. Strict cohort selection,
noisy correlated ten-attempt labels, evaluator bias, unverified learned fitting lineage,
shared contexts, prior reuse of these data and the single-model/configuration domain
limit all claims. This milestone supplies a reproducible sensitivity analysis and
leakage guards, not evidence that LUX has reduced hallucinations in deployment.
