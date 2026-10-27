# Research Decisions - LUX â€” Language Uncertainty Extractor

## 1. Project
**CONFIRMED**: LUX â€” Language Uncertainty Extractor

## 2. Research Question
**CONFIRMED**: Can combining observable prompt-level features with the target LLM's internal pre-generation hidden representations improve hallucination-risk prediction compared with using either information source alone?

## 3. Problem Statement
**CONFIRMED**: Hallucination-risk prediction for open-weight LLMs must be made *before* the target LLM generates its answer.

## 4. Research Motivation
**CONFIRMED**: Existing hallucination detection often relies on post-generation analysis. Predicting risk pre-generation allows for preventative measures and better understanding of model uncertainty.

## 5. Research Gap
**CONFIRMED**: The project investigates the comparatively less-explored combination of prompt-level features and internal hidden-state representations for pre-generation hallucination-risk prediction.

## 6. Proposed Contribution
**CONFIRMED**: A modular framework to extract and fuse prompt features and internal hidden states to predict the probability of hallucination.

## 7. Confirmed Architecture
**CONFIRMED**:
- QA Dataset Input
- Prompt Feature Extraction (Linguistic, Syntactic, Complexity, Entity, Semantic)
- Frozen Qwen3 (Target LLM)
- Hidden State Extraction (Internal Representations)
- Feature Fusion Layer
- Hallucination-Risk Predictor (Separate Model)

## 8. Ground-Truth Construction
**CONFIRMED**:
- Multiple independent generations from the frozen target LLM.
- Factual/semantic correctness evaluation for each generation (CORRECT/INCORRECT).
- Empirical Risk = incorrect generations / total generations.

## 9. Target Variable
**CONFIRMED**: Empirical hallucination risk âˆˆ [0,1].

## 10. Pre-Generation Constraint
**CONFIRMED**: Predictor inputs must NOT include the generated answers. The prediction must be possible before generation starts.

## 11. Confirmed Research Constraints
**CONFIRMED**:
- Target model family: Qwen3
- Qwen3 must remain frozen.
- Predictor is a separate model.
- Generated answers are NOT used as predictor features.

## 12. Candidate Datasets
**CANDIDATES / UNRESOLVED**:
- SQuAD
- Natural Questions
- TriviaQA
- HotpotQA

**STATUS**: The SQuAD training JSON is currently available locally, but this does NOT mean SQuAD has been approved as the final dataset. Final dataset selection is pending.

## 13. Unresolved Research Decisions
**UNRESOLVED**:
- Exact Qwen3 checkpoint (e.g., specific parameter count, version).
- Model and tokenizer revisions.
- Exact implementation of prompt features (specific parsers, NER models).
- Selection of semantic embedding model.
- Generation protocol (exact generation parameters and whether particular parameters are fixed or experimentally varied will be determined during the later ground-truth generation-protocol checkpoint).
- Factual/semantic correctness evaluation methodology and evaluator model.
- Exact risk-predictor model architecture.
- Final set of evaluation metrics (continuous and binary).
- Train/validation/test splitting strategy.

## 14. Decisions That Must Be Experimentally Determined
**EXPERIMENTALLY DETERMINED**:
- Hidden-state layer selection (which layer(s) provide the best predictive signal).
- Hidden state representation strategy (e.g., mean pooling, last token).
- Optimal feature fusion strategy.

## 15. Assumptions That Are Explicitly Forbidden
**CONFIRMED**:
- Do NOT assume generated answers can be used for prediction.
- Do NOT assume Qwen3 can be fine-tuned.
- Do NOT assume a specific layer for hidden states without experimental proof.

## 16. Checkpoint 1 Status
**CONFIRMED**: Completed. Repository structure established, research parameters codified in `experiment.yaml`.

## 17. Evaluator Versioning (CP4.5.0)
**CONFIRMED**: The independent evaluator is versioned as `gemini-3.8-flash`
with prompt provenance `CP4.5.0`. This change is due to Gemini 2.5 Flash
model-access restrictions across the configured projects. The evidence
restrictions, response schema, and RELIABLE / UNRELIABLE / INADEQUATE
definitions are unchanged. CP4.5.0 results are stored separately and must not
be mixed with prior evaluator versions when constructing targets.

**CONFIRMED**: A prompt receives an empirical risk target only when all ten
repeated generations have resolved CP4.5.0 RELIABLE or UNRELIABLE labels.
INADEQUATE, manual-review, failed, and missing evaluations are not counted as
binary correctness outcomes.

**EXPERIMENTAL STATUS**: Seven available CP4.5.0 prompt targets are complete,
and all have zero observed risk. Current predictor comparisons match the
constant training-mean baseline and do not demonstrate predictive
discrimination or improvement. Completing the versioned evaluation/predictor
engineering is not evidence that the research hypothesis is supported. A
different evaluator intended for primary analysis requires a new provenance
version and output directory, plus full-cohort reevaluation; its labels must
not be mixed with CP4.5.0. A different vendor additionally requires a
provider adapter.

**2026-10-17 EXECUTION UPDATE**: The requested primary policy uses
`gemini-2.5-flash` / `CP4.4.1`, with CP4.5.0 Gemini 3.8 results retained as
secondary. A persisted controlled evaluation received a global 404 stating
Gemini 2.5 Flash is no longer available to new users; the primary key also
returned 429 `RESOURCE_EXHAUSTED`. Therefore only 18 of 310 generations have
resolved historical CP4.4.1 labels, yielding one complete prompt target
(risk 1.0). The version-isolated partial target report is
`data/production/gemini_2_5_flash_CP4_4_1/targets.json`. It is insufficient
for a predictor comparison. Seventeen same-generation pairs agree between
CP4.4.1 and CP4.5.0 (8 RELIABLE, 9 UNRELIABLE); this small selected comparison
does not justify merging evaluator versions or claiming independent ground
truth.

## 18. Checkpoint 2 Gate
**SCOPED DECISION**: For the requested expanded cohort, use the local SQuAD
v2.0 training split. This does not select or exclude the other candidate
datasets for future external validation.

## 20. Expanded 500-Prompt Cohort — Preflight Only

The requested cohort selects 500 unique SQuAD v2 training prompts with ten
generations per prompt. Selection preserves the existing 31 unique prompts
and samples the remaining 469 without replacement using seed `20261017`.
Content duplicates are excluded by comparing Unicode NFKC-normalized,
case-folded, whitespace-collapsed question/context pairs. The selected prompt
IDs, canonical dataset indices, seed, source hashes, and method are recorded in
`data/production/squad_v2_expanded_500/selection.json`; measured planning
counts are in `data/production/squad_v2_expanded_500/preflight.json`.

Measured preflight found 130,319 canonical training scenarios, 130,219 unique
question/context pairs, and 100 duplicate-content rows. The 500-prompt cohort
contains 31 compatible existing prompts / 310 compatible generations and
requires 469 new prompts / 4,690 new generations. The 31 existing compatible
hidden-state artifacts will be copied into the isolated cohort directory;
469 hidden-state artifacts are still missing. Existing source generations,
hidden states, and evaluation records have not been modified. No Qwen
generation or evaluator call was started in this preflight task.

For the expanded primary target, use only `gemini-3.5-flash-lite` / `CP4.7.0`.
Adopt the stricter versioned target rule: a prompt is eligible only with ten
same-version `SUCCESS` labels, each `RELIABLE` or `UNRELIABLE`; risk is
`UNRELIABLE / 10`. Inadequate, failed, manual-review, missing, and incomplete
outcomes are excluded, not counted as reliable. This deliberately selects
the CP4.7.0 versioned builder rather than the legacy general builder's
partial-success denominator. CP4.5.0 and older evaluator outputs remain
separate. The preflight estimates 4,690 additional evaluation requests if all
new generations succeed and each resolves once; retries may add calls. Quota,
runtime, and cost were not probed or estimated.

Implementation, focused tests, and staged commands are documented in
`docs/expanded_500_prompt_cohort.md`. Predictor targets remain continuous;
baseline, prompt-only, hidden-state-only, and fusion comparisons use one
prompt-identity split and training-only preprocessing. No predictor result is
claimed until the expanded cohort has been executed and validated.

## 19. CP4.7.0 Evaluation and Predictor Results

**CONFIRMED**: A separate full-cohort evaluator version uses
`gemini-3.5-flash-lite` with prompt provenance `CP4.7.0`, stored under
`data/production/cp4_7_0_gemini_3_5_flash_lite/`. The model was selected after
the persisted Gemini 2.5 Flash-Lite retirement response suggested it. The
evidence instructions, response schema, and label definitions were retained;
the model/prompt version and output location are distinct from CP4.5.0. All
310 persisted Qwen generations were reused, with no new target-model
generation or hidden-state extraction.

The CP4.7.0 run resolved 310/310 latest records: 248 RELIABLE, 61 UNRELIABLE,
and 1 INADEQUATE. The inadequate record has no binary correctness label.
Thirty of 31 prompts have complete ten-generation binary targets; one prompt
is excluded because it contains the inadequate result. Across the 30 eligible
targets, risks are distributed as 22 at 0.0, one at 0.3, one at 0.6, two at
0.7, one at 0.8, and three at 1.0. CP4.7.0 labels remain isolated from all
other evaluator versions.

The saved execution log records completion with no missing generations. An
initial quota-exhausted evaluation was resumed after the configured cooldown;
the resumed run took 982.36 seconds, evaluated the remaining generations and
resolved the prior failed record. The history contains 320 API attempts across
311 evaluation iterations, including one recorded 429 quota-failed iteration
on `GEMINI_API_KEY_10`; its later successful retry is preserved separately.
The quota error was not assigned a label. No evaluation worker remains active.

Prompt-only, hidden-state-only, and fusion baselines used the same prompt-level
split (21 train, 9 test) and 30 eligible targets. Target risk spans 0.0–1.0
with a mean of 0.2033. Test MAE was 0.4355 for prompt-only, 0.4387 for
hidden-state-only, and 0.4519 for fusion; the corresponding training-mean
baseline MAE was 0.4450. MAE changes versus baseline were +0.0095, +0.0063,
and -0.0069 respectively. Paired bootstrap 95% intervals for those changes
were [-0.0229, 0.0364], [-0.0745, 0.0739], and [-0.1128, 0.0755]. The test
set is only nine prompts, all three models have negative R², and every
improvement interval includes zero. These results do not establish useful
predictive performance or support a fusion improvement.
