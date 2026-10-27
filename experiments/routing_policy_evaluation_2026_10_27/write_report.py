"""Render the complete routing report from results.json; exclusive output creation."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def pct(value):
    return 'undefined' if value is None else f'{100*value:.2f}%'


def render(d):
    text = '''# LUX Milestone 2: retrospective routing evaluation

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

'''
    text += f"N={d['n']}; zero-risk prompts={d['n_zero']} ({pct(d['n_zero']/d['n'])}); mean risk={d['mean_risk']:.10f}. "
    text += f"The eligible cohort contains {d['unreliable_evaluator_outcomes']} UNRELIABLE judgments out of 4,860 attempts.\n\n"
    text += '| Source | Exact path | SHA256 |\n|---|---|---|\n'
    for name, source in d['sources'].items():
        text += f"| {name} | {source['path']} | {source['sha256']} |\n"
    text += f"\nAnalysis SHA256: {d['analysis_sha256']}. Declared contract SHA256: {d['contract_sha256']}.\n"
    text += '''
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
'''
    for key, row in d['references'].items():
        m = row['aggregate']
        text += f"| {key} | {pct(m['escalation_fraction'])} | {pct(m['normal_empirical_risk'])} | {pct(m['captured_unreliable_fraction'])} | {pct(m['normal_any_unreliable_prompt_fraction'])} |\n"
    text += '''
## Complete threshold trade-offs (no selected policy)

All valid configurations and all declared thresholds follow. q=0 remaining is an
idealized per-original-prompt scenario, not an observed improvement. Full q=.5/q=1
results, per-fold distributions and paired intervals are in results.json.

| Configuration | Threshold | Escalated | Normal proxy risk / false reassurance | Captured mass | Any-unreliable normal prompts | q=0 remaining |
|---|---:|---:|---:|---:|---:|---:|
'''
    for row in d['policies'].values():
        m = row['aggregate']
        text += f"| {row['predictor']} | {row['threshold']:g} | {pct(m['escalation_fraction'])} | {pct(m['normal_empirical_risk'])} | {pct(m['captured_unreliable_fraction'])} | {pct(m['normal_any_unreliable_prompt_fraction'])} | {pct(m['hypothetical_remaining_per_original_prompt']['0.0'])} |\n"
    text += '''
At any strictly positive threshold the zero median allows every prompt; at zero it
escalates all. The training-fold means are each constant within their test fold and
cannot rank prompts within a fold. A low regression MAE therefore does not establish
routing utility. Learned curves may show retrospective concentration of risk, but
the complete tables and conditional paired intervals do not justify selecting them.

## Contract-excluded configurations

| Configuration | Invalid risk predictions (of 486) |
|---|---:|
'''
    for key, row in d['excluded'].items():
        text += f"| {key} | {row['invalid_predictions']} |\n"
    text += '''
## Execution, tests, review and preservation

Activation in every execution shell used process-only policy bypass, followed by
. .\\.venv\\Scripts\\Activate.ps1; sys.executable was verified against the project
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
'''
    return text


if __name__ == '__main__':
    results = json.loads((HERE/'results.json').read_text(encoding='utf-8'))
    with (HERE/'report.md').open('x', encoding='utf-8') as f:
        f.write(render(results))
