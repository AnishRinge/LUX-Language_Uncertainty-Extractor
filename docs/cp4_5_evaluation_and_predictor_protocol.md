# CP4.5.0 Evaluation and Predictor Protocol

## Evaluator change

The prior production evaluator was `gemini-2.5-flash` with prompt version
`CP4.4.1`. It returned model-access 404 errors across all ten configured
credential slots; the primary project also had a separate daily-quota 429.
CP4.5.0 therefore uses `gemini-3.8-flash` as the independent evaluator.
The change is versioned in every record as `evaluator_model=gemini-3.8-flash`
and `prompt_version=CP4.5.0`.

The evidence restrictions, label definitions, prompt instructions, and
structured response schema are unchanged from CP4.4.1. This version change
records the evaluator-model switch and is not intended to alter what counts as
RELIABLE, UNRELIABLE, or INADEQUATE. The target remains the frozen
`Qwen/Qwen3-1.7B` revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`.

## Data isolation and restart behavior

New records are written only under
`data/production/cp4_5_0_gemini_3_8_flash/evaluations/`; prior records under
`data/production/evaluations/` remain historical and are not reused as CP4.5.0
labels. The runner consumes the existing generation manifest and uses each generation
record's stored question, context, answer, and ID directly; it does not
reconstruct prompts from the canonical scenarios file. It does not call the
target model. Result JSON and the run log are atomically replaced. Matching
`SUCCESS` and `MANUAL_REVIEW` records are terminal checkpoints. A quota-failed
record can be retried after its 24-hour cooldown; the retry is written under
`evaluations/retries/` with the prior attempt history, leaving the original
failed record unchanged.
Credential slots that return project-access or quota failures are recorded by
slot name and excluded for subsequent generations of this evaluator version.
Quota blocks expire after a conservative 24-hour cooldown; project access
denials remain excluded. Secrets are never persisted.

## Provider boundary and resumable execution

The current runner is explicitly Gemini-specific: it constructs
`GeminiEvaluator`, supplies the Gemini prompt/schema builders, and uses
Gemini credential-slot and quota error semantics. The evaluator can accept an
injected Gemini-compatible client for tests, but that is not a provider-neutral
interface. Supporting another vendor requires selecting its model/API and
credential convention first, then adding a small adapter that implements the
same `evaluate(request) -> EvaluationRecord` contract with equivalent schema
validation, retry/error classification, redaction, and provenance. The runner
also needs a provider/version configuration that derives a distinct output
directory and record provenance. No alternate provider, model, credentials, or
quota are configured or verified here, so the Gemini runner must not be pointed
at another provider and no such calls have been made.

If CP4.5.0 Gemini quota becomes available after its cooldown, resume the
existing Gemini cohort with the project virtual environment:

```powershell
& .\.venv\Scripts\python.exe .\run_eval_cp4_5.py evaluate --spacing 1 --max-seconds 9000
```

This resumes within `cp4_5_0_gemini_3_8_flash`, checkpoints each result, skips
resolved records of that version, and continues missing generations. The
current credential state is quota-blocked; do not run it until recovery is
meaningful. Any selected second evaluator intended for primary analysis must
instead receive a new prompt/evaluator version and isolated output directory,
and must evaluate all 310 existing generations. Reusing generations is
required; CP4.5.0 labels must not be mixed with the new evaluator's labels.

## Empirical target

The generation manifest contains 310 successful generations in 31 prompt
identity groups (ten generations per prompt). A prompt receives an empirical
hallucination-risk target only when all ten records are CP4.5.0 `SUCCESS` with
binary `RELIABLE` or `UNRELIABLE` labels. The target is
`UNRELIABLE / 10`. `INADEQUATE`, `MANUAL_REVIEW`, failed, and missing
evaluations are reported but excluded; any prompt without ten eligible labels
has a null target. Each eligible target also reports a 95% Wilson interval for
the ten-generation Bernoulli rate. Evaluator failures never receive a
correctness label.

## Predictor comparison

When enough complete targets exist, prompt-only, hidden-state-only, and fusion
Ridge baselines use a common prompt-identity train/test split. Feature
standardization, PCA, and model fitting are performed on training prompts
only. Pre-generation prompt features and existing pre-generation hidden states
are used; generated answers, evaluation outputs, labels, and post-generation
fields are excluded from model inputs. Results include held-out MAE, RMSE, R²
when defined, the training-mean baseline, sample/target distributions, and
paired prompt-bootstrap percentile intervals for MAE and RMSE.

## Limitations

The evaluator is a model-based judge rather than independent human ground
truth. Switching evaluator versions may change labels despite retaining the
same instructions/schema. The SQuAD v2 dataset scope, 31 prompt identities,
ten stochastic generations per prompt, small prompt-level test set, and strict
complete-target rule constrain generalizability and power. Bootstrap intervals
over a small held-out set are descriptive, not a substitute for a larger
external validation. No model improvement should be claimed unless held-out
results support it.

## Engineering status and experimental findings (2026-10-17)

### Engineering completed

CP4.5.0 has a version-isolated, resumable evaluator that consumes the
persisted generation manifest, writes each result atomically, and preserves
prior records. Its strict target builder reports missing, failed, manual-review,
and inadequate outcomes separately. Prompt-only, hidden-state-only, and fusion
comparisons use one prompt-identity split, and predictor inputs exclude
generated answers and evaluator outputs. Mocked regression tests cover
versioning, resume/retry persistence, and target eligibility. This establishes
pipeline behavior, not predictive performance.

### Experimental findings and limitations

The new evaluator accepted actual generation-evaluation requests. The run
recorded 84 of 310 generation results before all ten configured credential
slots returned daily-quota exhaustion. The runner stopped rather than retrying
known-exhausted slots. The result set contains 82 `SUCCESS`, one
`MANUAL_REVIEW`, and one `FAILED` record; 226 generations have no CP4.5.0
record yet. Label counts are 70 `RELIABLE`, 12 `UNRELIABLE`, and one
`INADEQUATE` (the manual-review outcome). The failed record has no label.
There were 95 recorded API attempts: ten quota failures and two transient
attempts that were retried. The 9,000-second budget was not reached, and no
evaluation worker remains active.

The first run stopped at generation
`squad_v2_5709535eefce8f15003a7df5:31150:1001` because it reconstructed
question/context from the canonical file rather than using the stored
generation prompt. This implementation issue was corrected; the restart
reused the first 60 CP4.5.0 records and successfully evaluated further
generations from the manifest. Generation prompts, answers, and hidden states
were not regenerated or modified. All prior evaluator records remain in the
legacy directory.

Seven prompts meet the complete ten-label target rule; the other 24 targets
are null because labels are inadequate, manual-review, failed, missing, or
otherwise incomplete. All seven eligible risk targets equal `0.0`; each has a
95% Wilson interval of approximately `[0.000, 0.278]`. The available outcome
labels are therefore strongly selected by target eligibility, and the
empirical-risk distribution has no variance.

All three predictor baselines were run on the same prompt-identity split:
four training prompts and three test prompts. Prompt-only, hidden-state-only,
and fusion each report MAE `0.000`, RMSE `0.000`, and undefined R²; each
matches the training-mean baseline with no MAE improvement. Their paired
bootstrap intervals are degenerate at zero because every test target is zero.
These numbers do **not** demonstrate predictive discrimination or model
improvement. More quota capacity, fully evaluated prompt groups, and
non-constant empirical-risk targets are required before the research
comparison is interpretable.

At the latest inspection, all ten Gemini slots remained quota-blocked, the
24-hour cooldown had not elapsed, and no evaluation worker was active. No
additional live calls were made. The CP4.5.0 comparison is numerically
computable from seven complete prompts, but the target distribution is
constant; it provides no evidence that LUX predicts hallucination risk.

The comparison was rerun offline after the pipeline audit into a new,
non-overwriting directory,
`predictor_models/revalidation_20261017_1917/`. All three input alignments had
seven prompts; with random state 42, the common split remained four train /
three test prompts. Targets were all zero in both splits. Prompt-only,
hidden-state-only, and fusion each returned MAE 0, RMSE 0, undefined R²,
baseline MAE 0, and zero improvement; their 2,000-resample paired bootstrap
intervals were degenerate at zero. This confirms the prior computation, not
predictive capability. The predictor's minimum shared split is five common
complete prompts (three train, two test), but that size does not overcome
constant-target limitations. Target-builder and predictor mocked tests passed
(35 tests); no provider calls were made during validation.

### Historical Gemini 2.5 comparison

The older `data/production/evaluations/` records contain 18 resolved binary
labels for `gemini-2.5-flash` / `CP4.4.1`. Seventeen generation IDs overlap
with successful binary CP4.5.0 records. The labels agree on all 17 (8
`RELIABLE`, 9 `UNRELIABLE`). This selected overlap is descriptive agreement
only; it is not independent ground truth and the evaluator-version records
remain separate. See
`data/production/evaluator_agreement/gemini_2_5_flash_CP4_4_1__gemini_3_8_flash_CP4_5_0.json`
for paired generation IDs, labels, statuses, and source-record provenance.

The subsequent Gemini 2.5 attempt failed globally because the API responded
that `gemini-2.5-flash` is no longer available to new users. Therefore the
historical 18 labels do not constitute a completed 310-generation primary
cohort. A strictly version-filtered partial primary target artifact is saved at
`data/production/gemini_2_5_flash_CP4_4_1/targets.json`: it contains one
eligible prompt (risk 1.0) and 30 ineligible prompts. This is insufficient for
predictor comparisons; no primary-model predictors were trained.

### Flash-Lite candidate attempt

One controlled actual evaluation of existing generation
`squad_v2_5ad3e4e5604f3c001a3ff58f:69929:1001` was made using
`gemini-2.5-flash-lite` / `CP4.6.0`, with one credential, no retries, and no
failover. It returned 404 `NOT_FOUND`: Flash-Lite is no longer available to
new users. The API suggested Gemini 3.5 Flash-Lite; that model was not in the
approved candidate list and was not attempted. The failure, attempt slot, and
null label are persisted under
`data/production/cp4_6_0_gemini_2_5_flash_lite/`. Its target artifact has zero
eligible prompts. No full-cohort runner was started.

The saved Gemini 3.8 quota responses identify
`generate_content_free_tier_requests` with limit 5 for `gemini-3.8-flash`
across all ten slots. They do not provide a complete current RPM/TPM/RPD
dashboard view, and no quota probe was made. Therefore none of the user-listed
models is currently a viable full-cohort primary: Gemini 2.5 Flash and
Flash-Lite return global retirement errors, while Gemini 3.8 is exhausted in
the persisted run state.
