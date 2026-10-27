# Policy-selection lineage recovery audit

Date: 2026-10-27. Scope: read-only recovery and experiment design; no new model
fits, model predictions, policy evaluation, generation or evaluator calls.

## Decision

**Partial lineage recovered; valid joint model/threshold selection cannot be
recovered from the files found.** The important additional finding is that all 33
learned-model records contain five `selected_params_per_fold` entries: **165
recorded hyperparameter choices**, plus timing and failure fields. The six baseline
records have a smaller schema. Inspecting the first baseline record alone misses
this evidence. This supplements the earlier audits without changing their results.

The choices are real saved metadata, not reconstructed guesses. They do not include
the inner validation scores, inner prompt assignments, full search definitions,
fitted preprocessing lineage, or threshold-selection decisions needed to certify
the selection procedure. No model or threshold is recommended as a winner.

The minimum next experiment is a **new, explicitly authorized predictor-only nested
selection/evaluation run**, reusing the eligible targets, pre-generation features
and saved outer folds. Original answer generation and evaluation need not be repeated.
This audit stops before implementing or running that experiment.

## Search scope and method

Verified activation in the tool's PowerShell process using process-only execution
policy bypass, `. .\.venv\Scripts\Activate.ps1`, and an assertion that
`sys.executable` equals this repository's `.venv/Scripts/python.exe`.

Read the repository audit, implementation report, end-to-end validation report,
nested-CV report, verification report/JSON, earlier train/test and training-only-CV
reports, routing report/analysis, inference policy, predictor serialization code,
configuration, dependency declarations and the corresponding saved artifacts.
Searches included ignored/hidden repository files, the registered
`.kilo/worktrees/reminiscent-mouth` worktree, model/prediction filenames, Python
source, checkpoint schemas and resume logs. Local Git history was queried read-only
for CV-related filenames and Python/notebook additions containing
`selected_params_per_fold`, `GridSearchCV` or `nested_cv_results`; no original
expanded nested-CV runner was found there. No branch/worktree was changed.

The negative finding is limited to these local sources. External shell history,
another machine, remote storage or an original author's private execution environment
may retain artifacts; those locations were not accessed. No secrets or prompt text
were printed. Saved joblib files were inventoried and matched to JSON metadata, not
unpickled or executed.

## Actual evidence recovered

Paths below are relative to the repository root.

| Required evidence | Found and checked | Sufficiency |
|---|---|---|
| Outer train/test prompt IDs | `experiments/expanded_model_selection_2026_10_24/fold_assignments.json`: five records with `fold`, `train_pids`, `test_pids`, `train_rows`, `test_rows` | Membership verified; not proof of which data a search actually used |
| Fold summary | Same directory, `folds.json`: `outer_folds`, seed 12345 | Counts only; adds no inner lineage |
| Prediction-to-prompt and target mapping | `nested_cv_results.json`: 39 configurations with `oof_pids`, `oof_true`, `oof_predictions`, `fold_maes`, pooled MAE/RMSE/R² | All configurations align to the same 486 unique IDs and saved targets |
| Per-fold chosen hyperparameters | Same JSON: 33 learned records additionally contain `selected_params_per_fold`, `failures`, `times` | 5 entries per learned record; records choices but not how they were selected |
| Search checkpoint | `checkpoint.json`, same directory | Its `results_all` equals the final 39 result records exactly; no independent inner-prediction store |
| Global chosen result | `nested_cv_best.json`: `best`, `best_score` | `best` is an existing OOF result row and matches minimum pooled OOF MAE; not per-outer-fold family or policy selection |
| Execution progress/configuration hints | `resume.log`, `resume_opt.log`, `report.md` | Fold-numbered timings and modified-search/PCA notes; no validation prompt/score tables |
| Original nested-CV execution source | Not found in the searched source/history | Cannot verify full preprocessing and selection boundaries |
| Inner fit/validation IDs and predictions | Not found for this five-fold experiment | Required for threshold selection; ordinary pooled OOF is insufficient |
| Per-outer-fold threshold/action/cost selection | Not found | Milestone 2 deliberately did not perform this selection |
| Full preprocessing/calibration versions and fitted transforms per fold | Not found for the expanded nested-CV experiment | Generic feature-family/pooling notes do not recover fitted state |

Outer fold counts are `(train, test)` = `(388,98)`, then four times `(389,97)`.
Training/test IDs are disjoint, cover the cohort, and each ID occurs in exactly one
outer test fold. All **195 saved fold MAEs** (39 × 5) were recomputed from the
ID-aligned OOF rows and corresponding test IDs and matched within 1e-12. This checks
prediction alignment and the fold-metric ordering; it does not prove historical
training exclusion. Hyperparameter entries are positionally ordered lists, not
records carrying explicit fold IDs or fit IDs. Their intended fold association is
consistent with the surrounding fold metrics/logs, but no execution lineage binds
each choice to a specific fitted pipeline.

Examples of actual recorded choices:

| Configuration | Five entries in `selected_params_per_fold` (fold order) |
|---|---|
| `ridge|prompt` | alpha = 10, .01, 10, 10, 1 |
| `ridge|hidden` and `ridge|fusion` | alpha = 10 for all five |
| `rf|prompt` | (max_depth,n_estimators) = (5,100), (5,50), (10,100), (5,50), (5,100) |
| `rf|hidden` and `rf|fusion` | (5,100), (5,100), (5,50), (5,50), (5,100) |
| `gbr|hidden` | learning_rate=.1, max_depth=3, n_estimators=50, `_note=reduced_search_pca_applied` in all five |

All 33 learned records have empty `failures` arrays; this is a recorded absence of
failures, not proof that all search trials succeeded or that no failures were omitted.
The resume report says PCA(50) was applied to hidden features for tree models and
the `gbr|hidden` search was reduced. The exact full trial grids, scoring calls,
candidate scores, fitted scaler/PCA parameters, original library versions and
outer-specific inner split seeds are not saved in these result records. Do not
expand a chosen parameter value into an assumed original search grid.

### Other real artifacts that do not close this gap

**Earlier training-only CV.** `cv_run_proper.py` exists and describes a separate
340-train/146-test experiment. `experiments/cv_model_comparison_2026_10_24/` has
`split.json`, `cv_results.json`, `experiment_results.json`, `test_predictions.npy`,
`baseline_predictions.npy`, `test_y.npy`, `test_pids.npy` and reports.

The source applies GroupKFold(5) within the 340 training IDs, fits StandardScaler/PCA
inside each training fold, records five MAEs for each of 12 fixed configurations,
selects by mean training-CV MAE, then refits for the 146 test prompts. The saved
`selected` record identifies `rf_hidden` and agrees with the minimum saved CV mean.
The saved split equals `experiments/final_cohort_2026_10_24/split.json`; all 146 test
IDs/targets match the canonical saved arrays. This is specific source/metadata
evidence of a training-only selection design, not an inference from its test ranking.

However, the source saves fold counts and aggregate errors, not the individual
training-validation predictions or validation IDs. Its fitted models are local
variables, not persisted outer-fold pipelines. It supplies neither a threshold
selection record nor the five outer-fold validation sets needed now. Recomputing
GroupKFold membership from inputs/version would be a reconstruction, not recovery
of a saved assignment. Existing test predictions cannot be reused as both threshold
training data and unbiased testing. These test outcomes have also already been
inspected repeatedly. Earlier reports with damaged formatting are not stronger
evidence than the actual source and JSON/NumPy files.

**Older serialized predictors.** Twelve joblib files exist under
`data/production/cp4_7_0_gemini_3_5_flash_lite/`:
`predictors{,_batch_03,_batch_04,_batch_05}/{prompt_only,hidden_only,fusion}/`.
Each family has `metadata.json` and a named `*_model.joblib`; each parent has
`comparison_results.json`. The source save routines persist Ridge pipelines or
Ridge/PCA/scaler objects, not the expanded experiment's 165 fitted outer models.

| Parent | Saved train/test counts | Training-ID overlap with outer test folds 0–4 |
|---|---:|---|
| `predictors` | 21 / 9 | 1, 7, 1, 7, 5 |
| `predictors_batch_03` | 68 / 30 | 8, 16, 14, 16, 14 |
| `predictors_batch_04` | 67 / 29 | 6, 15, 12, 20, 14 |
| `predictors_batch_05` | 67 / 30 | 17, 12, 14, 11, 13 |

The three families in each batch carry the batch split. IDs are eligible-cohort
subsets with disjoint batch train/test sets; no batch training set equals a required
outer training set. Every batch training set intersects every saved outer test fold.
Thus these models cannot supply outer-test-excluding selection scores for the
existing folds. Metadata records Ridge alpha=1, seed 42 and hidden last-layer
token-mean/PCA strategy, differing from the expanded layer-mean experiment. Batch
comparison files contain aggregate metrics, not inner-validation prediction tables.
Six additional CP4.5 joblib files belong to an earlier evaluator cohort and are not
substitutes for the CP4.7 labels. No expanded-experiment fold model files were found.

**Configuration and verification.** `configs/experiment.yaml` still describes an
earlier specification with unresolved feature/layer settings and old labels.
`requirements.txt` is not an exact historical environment lock. The nested-CV
verification JSON has `checks: {}` plus recomputed metrics; it proves neither inner
selection nor absence of leakage. Current `src/predictor/inference.py` supports only
versioned constant artifacts; it contains no learned-policy training lineage.
Milestone 2's `results.json` explicitly says selection was not performed and records
descriptive threshold masks, not historical validation decisions.

## Why the gap cannot be repaired with a join or aggregate metric

An OOF score for a row in outer fold B may have been produced using outcomes from
outer fold A. When evaluating A, using those B rows to choose a threshold or model
can feed A's outcomes back into selection. Removing A's rows from the selection
table does not remove that indirect dependence. Recovered hyperparameters do not
reveal these score-producing fits or their selection inputs. A global OOF winner
also uses all outer outcomes and must not become a retrospectively claimed nested
winner. Neither saved ranking, matching MAEs, nor a file named "nested" proves the
required inner procedure.

Individual validation predictions cannot be recovered from MAE/RMSE/R²: many
prediction vectors give the same aggregates. Recreating them requires source,
splits, versions, model state or new fits. Present batch weights/metadata do not
provide the missing outer-model state. Therefore no additional routing evaluation
or fabricated inner-prediction table was produced in this audit.

## Recommended bounded evaluation design (proposed, not executed)

### Operational objective to freeze before any selection

Use a primary binary policy: **normal generation versus conservative escalation
with abstention from an unverified answer**. This can be evaluated as a coverage /
retained-proxy-risk trade-off using existing outcomes, without assuming verification
repairs an answer. Let C_u be the cost of an unreliable delivered answer and C_e the
cost of escalation/abstention including human burden, lost coverage, latency and
compute in comparable units. Minimize on inner validation only:

`mean[C_u * y_i * I(allow_i) + C_e * I(escalate_i)]`.

The deployment owner must supply C_u > 0, C_e >= 0 and any coverage/escalation budget
before the run. Values are a decision input, not scientific facts. No values or
validated thresholds are inferred from the 486 outcomes. If the costs are not
specified, retain a labeled sensitivity analysis and do not declare a selected
operational policy. The existing production policy's allow/verify/escalate interface
is unchanged.

A distinct verification action returning an answer would require an additional
assumed residual risk multiplier q and its operational cost. That objective would
include `C_u*q*y_i` for verified rows. Existing artifacts do not identify q or
verification effectiveness; any q scenarios remain hypothetical. Do not present
the two cost models as equivalent, count abstentions as correct answers, or claim
measured reliability gains from either simulation.

### Minimal candidate set and selection boundaries

1. **Freeze the protocol and inputs.** Keep the exact five saved outer folds and
   486 strict IDs. Pin hashes, source, versions, seeds, cost definition, action
   semantics, candidate list, threshold grid, tie-break and failure rules in a new
   manifest before any run. Validate all five prompt-feature columns against the
   pre-generation source/schema; prior array construction has provenance limitations.
   The saved prompt feature order is question_chars, context_chars, question_tokens,
   context_tokens, question_ends_with_question_mark. Use an explicit input allowlist;
   generated answers, evaluator fields and target values are separate outcome data.
   A full hidden-feature source reconciliation is a prerequisite for adding hidden
   candidates, not for the prompt-only minimal option.
2. **Use a small predeclared set.** Minimum useful learned comparison: fold-trained
   mean, median and one fixed bounded prompt-only regressor already represented in
   the repository. For concrete planning, the existing 100-tree prompt RF in
   `cv_run_proper.py` is a candidate specification, not a performance winner: freeze
   its complete parameters/current version and use one worker plus a fixed seed.
   No hyperparameter search is needed. RF averages bounded training risks; there
   is no need to turn failed features into zeros or clip invalid predictions. An
   optional extension adds fixed hidden-only and fusion variants, not all 33 search
   configurations. Selecting historical per-fold parameters without their lineage
   is not a substitute for specifying this new experiment.
3. **Work inside one outer training partition at a time.** Recommended minimal
   stability design: three inner folds, saved explicitly with prompt IDs and seeds,
   entirely within its 388/389 training IDs. Stratification on zero/nonzero targets,
   if used, is computed from that outer training partition only. No outer test
   target is passed to preprocessing, fit, calibration or selection functions.
   No prompt's ten outcomes are split into train and validation as separate rows.
4. **Fit complete candidate pipelines on inner training rows only.** Any imputation,
   scaling, PCA, feature selection or hyperparameter selection must live inside that
   fit. Deterministic row-local extraction/layer pooling from saved pre-generation
   vectors may be reused; learned PCA/scalers fitted on the full cohort may not.
   Prefer rejecting missing/incompatible features in the new runner until an
   explicitly trained imputer is part of its schema. Each inner validation row
   gets scores from a pipeline that excluded its labels and all outer test labels.
5. **Select the joint model/policy pair using pooled inner validation scores only.**
   Evaluate the previously declared threshold grid/costs and explicit always-allow
   and always-escalate candidates. Baselines use inner-training means/medians.
   Pick the minimum declared cost subject to declared budget/coverage constraints;
   use a deterministic predeclared tie-break (e.g. simpler pipeline, then fixed
   candidate/threshold order). Record every candidate score and rejection, not only
   the winner. Do not optimize MAE as a surrogate for the actual routing objective.
6. **Refit the selected pipeline on that outer training partition only**, using its
   frozen settings; freeze its chosen threshold/actions. Evaluate it once on that
   outer test partition. A selected baseline needs only a constant calculation.
   The finite-sample score shift from an inner-trained to an outer-refitted model is
   a limitation to be measured by this outer test, not a reason to retune on it.
7. **Aggregate the five independently selected policies**, rather than choosing a
   global winner from their outer results. Persist all fold decisions, predictions,
   actions, IDs and metrics. The outer evaluation estimates the predeclared selection
   procedure. It does not itself deliver a globally chosen unbiased final model.
   A later full-data deployment fit uses the frozen selection procedure; its own
   performance requires independent confirmation.

The same rules apply to any internal early-stopping or tuning mechanism: its
validation rows must come only from its current training partition. Do not reuse
the ordinary cross-outer OOF table for any of these selection steps.

### Calibration and uncertainty

**Disable calibration in the minimum experiment.** Being bounded does not establish
calibration. With only 388/389 outer-training prompts and many zeros, a calibrator
adds another poorly supported estimation stage and is not necessary to test ranking
or cost-based threshold selection.

If later justified, one explicit leakage-safe alternative divides each outer
training set into disjoint A/B/C subsets: fit a base pipeline on A, fit a candidate
calibrator using A-model predictions/outcomes on B, and choose model/calibration
variant/threshold on C. Freeze that whole pipeline for outer testing. Do not refit
the base estimator on B/C and silently retain a calibrator trained for a different
fit. More data-efficient nested cross-fitting is possible, but needs its own fit
lineage and computation budget and is outside the minimum counts below.

Targets are noisy fractions from ten evaluator judgments; they are not latent true
risks. Keep integer n_unreliable/n_reliable and denominators in manifests. Repeated
attempts may be correlated, so do not treat 4,860 outcomes as independent prompts.
The 367 zeros (75.51%) make a zero-median MAE deceptively favorable without providing
risk ranking; no class reweighting/oversampling is introduced silently. Report
coverage/capture/false reassurance as well as MAE, RMSE and R² on identical prompts.

Use paired prompt-level uncertainty intervals, preferably within saved outer folds,
and report per-fold distributions and pooled count-weighted rates. Fixed-score/mask
bootstrap intervals are conditional and omit retraining/selection variance. If
contexts are shared, report that dependence; a separate context-grouped outer-split
experiment would be needed for stronger topic/context generalization. It must have
a new split manifest, not overwrite these folds. Prior repeated inspection of all
486 outcomes prevents calling this a pristine confirmatory dataset even when the
new computational selection boundaries are correct.

### References, metrics and required saved lineage

Always evaluate normal generation without a risk estimator and non-predictive hash
routing with a fixed salt and declared budgets. Include fold-trained mean and median.
A candidate's count-matched hash reference may use its escalation count within a
test fold, never the outcomes of that fold to choose which rows are escalated.

Report escalation rate, normal-path coverage, mean empirical risk among allowed
prompts, fraction of allowed prompts with any UNRELIABLE attempt, captured unreliable
mass, declared cost, and latency/feature-extraction burden where measured. Label
normal-path mean risk as a generation-level proxy for false reassurance; do not
confuse it with the fraction of prompts having any unreliable attempt. All-escalated
policies have undefined normal-path risk, not zero. Hypothetical post-verification
rates must state q and the denominator; actual downstream improvement remains unknown.

For every score, save prompt_id, outer_fold_id, inner_fold_id (where relevant),
pipeline_fit_id, source/feature hashes, complete preprocessing/model parameters,
fit and selection input ID lists, prediction and schema/pipeline version. For every
decision save costs/action assumptions, threshold grid, validation objective for
each candidate, tie-break, chosen model/threshold, seeds and timing. Separate
outcome tables from score/feature inputs; join outcomes only for authorized fitting,
selection and evaluation stages. Save selected model/preprocessing artifacts and
the environment lock. Assert that the union of **all label-bearing fit/selection
inputs**, not just final estimator fit IDs, excludes the corresponding outer test.

## Minimum computation and resource feasibility

No computation in this section has been started. Counts assume fixed candidate
settings, no calibration and no hyperparameter grid; changing those assumptions
changes the budget.

| New protocol | Learned inner fits | Selected outer refits, maximum | Total learned fits, maximum |
|---|---:|---:|---:|
| Absolute low-compute variant: one learned candidate, one fixed inner train/validation split per outer fold | 5 | 5 | 10 |
| Recommended minimum: one learned candidate, three inner folds per outer fold | 15 | 5 | 20 |
| Optional prompt/hidden/fusion candidates, three inner folds | 45 | 5 | 50 |

The one-split variant is valid if fully confined to outer training, but its roughly
78 validation prompts under an 80/20 split provide little positive-risk support.
Three-fold inner selection uses all 388/389 training prompts as validation once,
with fits on about 258–260 prompts, and is the recommended trade-off. Actual outer
refits are fewer if some folds select a baseline. Inner baselines need only mean /
median calculations; normal/hash policies need no learned fits. Threshold searches
on saved new validation scores require no additional estimator fits. Training-only
pooled validation score counts are 1,944 per learned candidate across outer folds;
there are 486 selected outer-test predictions. These rows do not yet exist.

Existing weights cannot eliminate these fits: the needed inner fits must exclude
the correct validation and outer-test IDs, and the needed final outer pipelines
are not persisted. Neither 165 hyperparameter dictionaries nor batch-specific
joblib models contain those weights. This is new predictor computation, not
"recovering" predictions from aggregates.

**Reusable inputs:** `experiments/final_cohort_2026_10_24/{X_prompt.npy,X_hidden.npy,
y.npy,pids.npy}`, the strict `targets_batch_05.json`, canonical prompts/source
metadata for feature reconciliation, raw saved hidden states if hidden candidates
are authorized, and the exact saved outer-fold assignments. Do not concatenate the
cumulative target snapshots. Existing mean/median outer OOF constants can be used
as verified references; new inner constants must be calculated from each inner fit
partition. Existing learned OOF scores remain historical diagnostics, not the new
threshold-selection data. No target-model/tokenizer network download is necessary.

**Memory/input sizes:** X_prompt is (486,5) int64, 19,440 data bytes; X_hidden is
(486,29,2048) float64, 230,916,096 data bytes (220.22 MiB). A deterministic layer-mean
copy is (486,2048), 7,962,624 bytes (7.59 MiB); additional scaler/PCA/fitting copies
and solver workspaces are not included. NumPy mmap can avoid loading the whole raw
array for the prompt-only minimum. The documented host has eight logical Intel64
processors and about 15.79 GiB physical RAM. Serial, one-worker fitting avoids
unmeasured process multiplication; no target LLM or GPU is needed. Peak fitting RAM
and output model sizes have not been measured and are not guaranteed by input sizes.

**Historical timing evidence, not new benchmarks:** saved learned-record `times`
sum to 4,058.46 s, but the missing runner prevents identifying the precise timing
scope. RF prompt records are 2.08–3.60 s per outer record (13.91 s total); RF hidden
59.83–104.63 s (452.73 s total); RF fusion 75.67–116.86 s (465.73 s total).
GBR fusion totals 1,310.93 s; HGB fusion 780.39 s. Resume logs contain aborted slow
work and changed searches. These are not comparable per-fit benchmarks and must
not be multiplied as if every timing represented one estimator fit. Milestone 1's
4.27 s median cold start / approximately 117 MiB process peak characterize constant
inference imports, not training or hidden extraction.

Consequently, exact wall-clock time/peak RAM for the proposed runner cannot be
estimated defensibly from saved timing records alone. The work is bounded by the
fit counts above; the prompt-only option is much smaller in input dimensions and
does not need PCA or hidden-state work. After explicit authorization, first time
one fixed prompt-only inner fit with its intended parameters, measure process peak
RAM, then project the remaining fit count with stated headroom and a user-approved
time/resource cap. Do not launch the 33-configuration historic search to answer
this narrower policy question. Storage for a few thousand numeric score rows is
small relative to existing arrays; full ID lineage and fitted-model sizes must be
included in the pilot's storage measurement. No paid service cost is required.

## Verification, preservation and next authorized action

Read-only checks passed: all seven inputs hashed in the routing results still match;
486 strict risks match binary count ratios; five outer test sets provide exact-once
coverage with disjoint training; all 39 OOF ID/target mappings align; 195 outer fold
MAEs match; checkpoint/final records agree; 165 selected-parameter entries are
present; earlier 146 test IDs/targets and 340/146 split are reconciled. No prediction
method or training runner was invoked. Tests were not rerun: the only change is
this report, and launching a legacy runner could fit models or overwrite outputs.

Initial historical inventory: 10,499 files under data/ and experiments/, with
SHA256 of the sorted (path,size,mtime_ns) inventory
`10ea743b4c69ac685a5d4858b2b210c33c18b037af11ae65521da1ad0955b4d1`.
The final inventory recheck matched exactly, and `git diff --check` passed. This is a metadata
preservation check, not a hash of every historical byte; seven direct source hash
checks add byte-level verification of the analyzed inputs. The working tree was
already dirty; no existing file/result was edited, no files were staged, and no
commit/push occurred.

Audit evidence fingerprints:

| File | SHA256 |
|---|---|
| Expanded nested_cv_results.json | 139772c8cf2ba821c2cc97add1bad96da5c08c337f41e38465df2f45349b5632 |
| Expanded fold_assignments.json | 49ca50c6a6d1c5a0cfffa155234a642b7078173a989051061d71eac2c8ca5f14 |
| cv_run_proper.py | 95c2331083dba223f07c6ad169091e98e4b10a53f85a4930c418cc70620efd87 |
| Earlier CV experiment_results.json | 8cbcdd875dd6edd3349fb7c6216085dd30dfc18fd9806b7ccae8cc2275d110bc |
| Earlier CV split.json | 72c56da23fc29946ab00d719547dbab10b731f42ff74c94a457c7d937df91008 |

**Next action authorized by the current request:** finish this audit/report and
stop. Do not run the proposed experiment. A subsequent instruction must authorize
predictor fitting and settle the cost/action definition, minimal versus extended
candidate scope, calibration exclusion and resource cap. Alternatively, inspect
additional original inner artifacts if the user supplies their location. Until
then, keep the production baseline fallback and Milestone 2 descriptive status;
do not promote a learned model or threshold.
