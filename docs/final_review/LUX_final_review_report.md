# LUX: Language Uncertainty Extractor

## Final academic review report

**Research topic:** Estimating the unreliability risk of a frozen language model before answer generation.

**Review baseline:** Repository and saved artifacts inspected on 27 October 2026.

**Status:** Completed empirical research and retrospective analyses, with a tested baseline-only Python inference interface. Learned-model deployment, calibrated individual risk estimates and demonstrated downstream reliability improvement remain unestablished.

This report distinguishes what was implemented, what was measured, what the saved evidence cannot establish, and what is proposed. Numerical tables are taken from the corresponding saved artifacts. No model fitting, answer generation, evaluator calls or new experiments were performed to prepare this document.

## Contents

1. Abstract and project overview
2. Problem statement and research motivation
3. Research methodology and dataset construction
4. Complete architecture
5. Feature engineering and hidden-state representations
6. Models, baselines and evaluation methodology
7. Results and interpretation
8. Implementation and software engineering
9. Limitations and unresolved issues
10. Contributions and novelty
11. Future work
12. Final conclusion
13. Five-minute presentation script
14. Examiner questions and model answers

An evidence register and final verification note follow Chapter 14.

# Chapter 1: Abstract and project overview

Language models can produce fluent answers whose claims are unsupported by the available evidence. Checking every completed answer may be costly. An estimate of risk before generation could help a system decide when to seek grounding, reserve additional verification, or decline to answer. The useful question is whether such an estimate contains predictive information, rather than whether it merely looks plausible.

LUX investigates whether observable prompt properties and a frozen target model's internal response to the prompt can predict empirical answer unreliability. The target model is `Qwen/Qwen3-1.7B`, pinned to revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`. The completed production cohort contains 500 SQuAD v2.0 prompts and ten sampled generations per prompt. A separate Gemini evaluator assigns evidence-grounded labels. A strict target is retained only when all ten evaluations provide successful binary reliability judgments.

The resulting dataset contains 486 eligible prompts. Empirical risk is the fraction of the ten generations labeled `UNRELIABLE`. Its mean is 0.208642, its median is zero, and 367 prompts, or 75.51 percent, have zero observed risk. Prompt-only, hidden-only and fused predictors were compared with simple constant baselines through several distinct evaluation stages.

The expanded benchmark did not demonstrate that learned models outperform the median baseline on out-of-fold mean absolute error. The median achieved MAE 0.208642; the lowest learned MAE was 0.214182 for prompt-only support vector regression. Some hidden-state models improved RMSE and R² relative to the mean baseline, but those differences do not establish dependable individualized prediction. A subsequent analysis assessed 176 fixed-threshold routing policies. It remained descriptive because the inner-validation predictions and complete selection lineage required for joint model and threshold selection were unavailable.

The supported inference interface therefore uses explicit, versioned cohort baselines. It validates inputs, artifact integrity and the target configuration, returns provenance and limitations, and exposes conservative routing. This is an engineering contribution, not evidence of a successful hallucination detector. My central conclusion is that LUX has produced an inspectable experimental foundation and a useful negative result, while the original hypothesis remains unconfirmed.

# Chapter 2: Problem statement and research motivation

## 2.1 What unreliability means here

Hallucination is often used broadly to describe invented or unsupported content. LUX uses a more specific operational target: an evaluator-established, material reliability failure relative to the supplied context and task. An answer can be unreliable because its central claim conflicts with the passage or materially goes beyond the evidence. A response can also be inadequate without enough evidence to establish a material failure.

This distinction matters. An incomplete answer is not automatically a hallucination, and an unverifiable claim is not automatically a false claim. The evaluator instructions explicitly distinguish supported, contradicted, unsupported, unverifiable and not-applicable claims. They also distinguish material failures from minor phrasing or formatting problems.

Consequently, LUX predicts a particular evaluator-derived unreliability construct. It does not measure every possible meaning of factual error, usefulness, harmfulness or open-world truth. A reliable answer under the passage-grounded instructions is not a universal guarantee of correctness.

## 2.2 Why predict before generation?

A post-generation verifier has access to the actual answer. It can compare specific claims with evidence, but only after the generation cost has been paid and the claims exist. A pre-generation predictor sees the question, context and potentially the target model's prompt-conditioned internal states. It must estimate the risk of a future stochastic answer without inspecting that answer.

That restriction makes the task harder and potentially useful. If a predictor could identify risky prompts at acceptable cost, a system could decide in advance to retrieve evidence, change the answering workflow, allocate checking resources or abstain. However, LUX must demonstrate that its ranking and decisions improve on simple references before those uses are justified.

The target is also different from predicting whether one particular answer will be wrong. Ten attempts provide a prompt-level estimate of how often the frozen configuration produced evaluator-labeled unreliability. They do not identify the outcome of the next attempt.

## 2.3 Why investigate internal representations?

Prompt length and question structure may reflect difficulty, but they say little about whether the target model can use a particular passage. Two questions of equal length can require very different reasoning. Surface features could also capture accidental properties of SQuAD rather than transferable difficulty.

Hidden states offer a different source of information. During a forward pass, the model transforms the prompt into contextual representations shaped by its own parameters. These may encode relationships, ambiguity or evidence compatibility that length counts miss. That is the motivation for investigating them, not a claim that a hidden vector directly represents factual confidence.

High dimensionality creates a competing explanation. With only 486 prompts, a predictor can fit corpus-specific structure or noise. Layer pooling may discard the relevant information, and a signal learned for one model revision may not transfer to another. The experiment therefore compares separate feature families and simple baselines rather than assuming hidden states must help.

## 2.4 Research question and hypothesis

**Research question:** Can combining observable prompt-level features with a frozen target model's pre-generation hidden representations improve prediction of empirical answer unreliability compared with either feature source alone and with constant baselines?

**Hypothesis:** Model-conditioned representations contain useful information beyond the implemented surface features, and fusion can exploit complementary signals.

The current evidence does not confirm this hypothesis as a dependable practical system. A favorable value on one error metric is insufficient. Predictive error, calibration, ranking, decision cost and operational overhead answer different questions.

Potential future applications include selective retrieval in a retrieval-augmented generation system, prioritizing verification, conservative answering in evidence-grounded assistants, and monitoring shifts in prompt difficulty. These are proposed applications. No live application benefit has been demonstrated.

# Chapter 3: Research methodology and dataset construction

## 3.1 Prompt source and normalization

SQuAD v2.0 provides questions paired with passages, including questions that cannot be answered from the passage. This supports a controlled study of evidence-grounded answering and abstention. Its reference structure is useful for dataset administration, but the ground-truth answerability flag is not a legitimate predictor feature in this experiment.

`src/dataset/squad_adapter.py` validates the raw structure and produces canonical scenarios with stable identifiers, prompt content, reference information and provenance. Answerable scenarios expect an answer. Unanswerable scenarios expect abstention from an answer unsupported by the supplied context. Reference fields are kept separate from prompt inputs.

The saved expanded-cohort selection manifest records 130,319 canonical scenarios, 130,219 unique normalized question-and-context pairs, and 100 duplicate-content cases in the source inventory. The expanded sample preserves 31 existing prompts and samples the remaining prompts without replacement using selection seed 20261017. The duplicate rule normalizes Unicode, case and whitespace. These steps reduce exact content duplication; they do not prove independence between related passages or topics.

The 500-prompt sample is a specific retained-and-expanded cohort, not a fresh uniform sample of all real-world questions. SQuAD's controlled passage setting also differs from open-domain queries.

## 3.2 Frozen target model and generation protocol

The model and tokenizer revision are locked. The generation code uses evaluation mode and a frozen target rather than fine-tuning Qwen for the risk-prediction task. A separate regressor learns from its prompt representations and empirical outcomes.

| Setting | Recorded value |
|---|---|
| Target model | `Qwen/Qwen3-1.7B` |
| Model revision | `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` |
| Execution device in completed production generation | CPU |
| Thinking mode | Disabled |
| Sampling | Enabled |
| Temperature | 0.7 |
| Top-p | 0.8 |
| Top-k | 20 |
| Maximum new tokens | 512 |
| Attempts per prompt | 10 |
| Generation seeds | Integers 1001 through 1010 |

`run_cp4_2_pilot.py` defines the seeds and rendering conventions reused by `run_production.py`. The prompt is rendered using Qwen's chat template with `add_generation_prompt=True` and `enable_thinking=False`. The user content contains the passage followed by the question. It does not expose the reference answerability label.

Using one revision and one sampling configuration makes the target definition more reproducible. It also limits the claim: changing the model, decoding settings, prompt rendering or thinking mode may change the outcome distribution. Seeds support repeatability within an environment but do not guarantee bitwise equivalence across library and hardware changes.

The choice of a 1.7-billion-parameter open model made local forward-state inspection feasible within the project's CPU-based workflow. This is a practical research choice. It is not evidence that the model is representative of all language models or optimal for this task.

## 3.3 Why ten generations?

One generated answer provides a single stochastic observation. Repeated attempts allow an empirical fraction to vary from zero to one. Ten attempts were the locked practical protocol, balancing outcome coverage against local generation and evaluator cost. The repository does not establish that ten is statistically optimal.

Ten attempts give coarse resolution: one changed judgment moves the target by 0.1. A zero target means that no unreliable outcome was observed in those attempts. It does not establish zero latent risk. Outcomes can also be correlated because they share the prompt, target model and generation procedure.

## 3.4 Evaluator protocol and labels

The final production target snapshot records evaluator `gemini-3.5-flash-lite` and prompt version `CP4.7.0`. The versioned wrapper in `src/evaluation/prompt_cp4_7.py` reuses the passage-grounded instructions in `src/evaluation/prompt.py`. Earlier evaluator versions exist in the repository but must not be mixed into the final cohort's labels.

The evaluator receives the question, supplied context and generated response. It does not receive target hidden states, generation seeds or target confidence as evidence for its judgment. Its claim assessments and generation-level decisions are saved separately from the predictor features.

| Label | Operational meaning |
|---|---|
| `RELIABLE` | No material reliability failure is established, and the response adequately addresses the task |
| `UNRELIABLE` | At least one material reliability failure is established |
| `INADEQUATE` | The response is incomplete or problematic, but a material reliability failure cannot be established confidently |

Technical failures, missing results and unresolved review states are not converted into unreliable-answer labels. The implementation separates transport or processing status from the substantive judgment.

## 3.5 Strict target construction

For prompt i, let u_i be its number of eligible generations labeled `UNRELIABLE`. The strict target is:

```text
y_i = u_i / 10
```

This target is defined only when exactly ten matching-version evaluations are successful and all ten labels are either `RELIABLE` or `UNRELIABLE`. Any inadequate, failed, missing or manual-review outcome makes that prompt ineligible under this protocol.

The completed cohort has 5,000 successful generation records and 5,000 evaluation files. The final snapshot contains 3,895 `RELIABLE`, 1,043 `UNRELIABLE` and 62 `INADEQUATE` judgments. Fourteen prompts contain the inadequate judgments and are excluded. Those 14 prompts also contain 49 reliable and 29 unreliable judgments, so their entire 140-generation contribution is excluded from the strict target dataset.

This leaves 486 prompts and 4,860 binary judgments, comprising 3,846 reliable and 1,014 unreliable judgments. No failed, missing or manual-review outcomes occur in the final excluded-prompt count summary. Thus, in this snapshot, the 14 exclusions are specifically associated with inadequate judgments, rather than absent generation work.

Strict exclusion avoids silently treating ambiguous evaluations as either correct or incorrect. It also changes the population being studied. Difficult or ambiguous prompts may be disproportionately excluded. The results describe the eligible cohort, not all 500 prompts without qualification.

`targets_batch_05.json` is the final cumulative snapshot. Earlier target batches overlap it. Summing all batch rows gives 1,231 historical rows, not 1,231 unique independent prompts. The correct final analysis uses the single cumulative snapshot and stable prompt IDs.

## 3.6 What the labels can establish

The targets are proxy measurements produced by another model. Evaluator bias, inconsistent materiality decisions and evidence interpretation can affect them. Repetition does not remove shared evaluator bias. The study therefore asks whether pre-generation signals predict these recorded empirical judgments, not whether it has discovered an independently verified probability of factual truth.

Independent validation could later estimate evaluator disagreement or bias. It is optional future work, not a missing mandatory execution step for the current software. The report makes no claim that human validation has been completed.

# Chapter 4: Complete architecture

## 4.1 Architecture diagram

The architecture is shown as plain text so that it remains readable in Markdown viewers without a configured Mermaid renderer. Two separate paths prevent the offline labels from appearing to be available during online prediction. Numbers refer to the detailed stages immediately below.

```text
OFFLINE RESEARCH AND TRAINING PATH

Canonical question and context
           |
     [1] Prompt validation and stable IDs
           |
     [2] Frozen target identity and generation settings
           |
           +------> [3] Five prompt features ------------------+
           |                                                  |
           +------> [4] Qwen prompt-only forward pass          |
           |             |                                    |
           |        Per-layer token-mean vectors               |
           |        Select last layer OR average layers        |
           |             |                                    |
           |             +------------------------------------+
           |                                                  |
           |                                      [5R] Fit preprocessing
           |                                           and predictor
           |                                           inside training folds
           |                                                  ^
           +------> [10] Ten generations -> evaluator labels    |
                           -> strict empirical targets --------+
                              SUPERVISION ONLY

     [5R] -> [6R] Held-out predictions -> regression and routing analyses
                          |
                 Experimental evidence and limitations
                          |
          No learned model is promoted to the supported online path

SUPPORTED ONLINE PATH: EXPLICIT CONSTANT BASELINE ONLY

Prompt text or exact question/context mapping
           |
     [1] Validate input
           |
     [2] Check target model, revision and configuration
           |
     [5I] Load and validate versioned baseline JSON artifact
           |
     [6I] Return its constant mean or median risk
           |
     [7] Structured output, bounds, provenance and warnings
           |
     [8] Configurable allow, verify or escalate decision
           |
     [9] FUTURE integration: generation, retrieval, verification, abstention

Stages [3] and [4] are NOT executed by the supported constant interface.
Offline targets, evaluator labels and generated answers do NOT enter online inputs.
```

The online diagram is a logical data-flow description. In the public API, the caller explicitly loads the artifact before calling `predict_risk`, and the prediction function revalidates the supplied artifact. This separation makes fallback selection visible rather than automatic.

## 4.2 Stage 1: Prompt input and validation

**Inputs and processing.** Offline research starts with canonical question and context fields and stable prompt identifiers. Online inference accepts either nonempty rendered text or a mapping containing exactly `question` and `context`. Each accepted field must be a string with non-whitespace content and at most 100,000 characters.

**Output and purpose.** The result is a valid prompt structure for the identity checks and, in the research path, feature extraction. Validation avoids treating missing data as a valid low-risk prompt.

**Implementation and failure behavior.** Canonical dataset validation is implemented in the dataset adapter and production cohort helpers. The public validation is in `src/predictor/inference.py`. Unknown mapping keys, malformed values and empty inputs raise errors. A generation record containing answers or labels is rejected rather than interpreted as a feature record.

**Status.** Implemented. The character limit is a resource guard, not a scientifically validated detector of out-of-domain prompts.

## 4.3 Stage 2: Target identity and configuration checks

**Inputs and processing.** The public request provides model name, revision and the full generation configuration. These are compared with the single supported frozen specification. Numeric and Boolean types are checked strictly.

**Output and purpose.** A compatible request proceeds to artifact-backed prediction. This prevents an estimate learned for one target configuration from being silently attributed to another.

**Implementation and failure behavior.** `GENERATION_CONFIG`, `MODEL`, `REVISION` and `_validate_config` implement the online contract. Missing settings, unknown settings and unsupported values raise errors. The research generation artifacts separately record the target settings actually used.

**Status.** Implemented for one target specification. Supporting another revision or decoding policy is future work requiring evidence, not merely a new display name.

## 4.4 Stage 3: Pre-generation prompt features

**Inputs and processing.** Research feature construction reads the question and context, counts characters and tokenizer tokens, and records whether the question ends with a question mark. Reference answerability and generation outcomes are not inputs.

**Output and purpose.** Five ordered features provide a low-dimensional comparison with the hidden representations. They connect to fold-specific preprocessing and regression in Stage 5R.

**Implementation and failure behavior.** Production feature builders, `src/predictor/baselines.py` and the final-cohort construction scripts establish the feature conventions. Some legacy helpers silently replace missing numerical features with zero. That is a confirmed research-code limitation, not behavior endorsed by the new interface. Schema and join auditing are still needed before learned deployment.

**Status.** Implemented for research. Deliberately absent from the supported constant inference path, which has an empty feature schema.

## 4.5 Stage 4: Hidden-state extraction and pooling

**Inputs and processing.** The frozen Qwen model receives the rendered prompt in a forward pass under `torch.no_grad()` before generation. `run_production.capture_hidden_states` saves each layer's last-token vector and mean across prompt-token states. No generated answer is required.

**Output and purpose.** The raw file provides 29 vectors of 2,048 values for each pooling form. A research consumer either selects the last layer or averages the layer axis, depending on the experiment. This creates model-conditioned features for Stage 5R.

**Implementation and failure behavior.** JSON artifacts record model revision, layer count, hidden size and sequence length. The last-layer extraction helper rejects absent or incorrectly shaped mean arrays. Some older loaders skip malformed files, so missingness must be audited rather than treated as safe zero features.

**Status.** Extraction is implemented and saved. Representation conventions differ between research paths. The baseline interface neither loads Qwen nor supports hidden-state inference.

## 4.6 Stage 5: Transformation, fitting and artifact loading

**Inputs and processing.** Offline learned pipelines receive feature matrices and training targets. Their scalers, PCA and regressors must be fitted only on the relevant training partition. Online inference instead receives a previously exported constant JSON artifact with its identity and source fingerprints.

**Output and purpose.** Research produces fitted predictors and held-out predictions. Online loading produces a validated baseline object that Stage 6I can use without feature extraction.

**Implementation and failure behavior.** The corrected CV source uses fold-local transformations. The expanded benchmark's full inner lineage remains unverified. `Artifact.load` checks schema version, pipeline version, identity, configuration, finite bounds, provenance hash format and the artifact fingerprint. Missing files and corrupted or incompatible objects raise exceptions. JSON duplicate keys are rejected; no executable model deserialization is needed for the baseline.

**Status.** Implemented and tested for baseline loading. Learned research fitting exists, but a standardized, verified learned deployment artifact is not the supported interface.

## 4.7 Stage 6: Risk prediction

**Inputs and processing.** Research regressors transform held-out feature rows into scores. Supported online inference reads the validated artifact's constant value. The mean artifact represents cohort expected risk; the median artifact represents the cohort median under an absolute-error objective.

**Output and purpose.** `predict_risk` returns a `Prediction` object carrying a risk value and identifiers for Stage 7. It does not inspect or generate an answer.

**Implementation and failure behavior.** `src/predictor/inference.py` accepts only an explicit `Artifact`. It revalidates nested artifact contents before returning. An unavailable estimator is not automatically replaced with a zero-risk result or a learned model. Caller-selected baseline fallback is explicit.

**Status.** Implemented for constants. No learned model is promoted, and a constant cannot discriminate between different accepted prompts.

## 4.8 Stage 7: Output validation and provenance

**Inputs and processing.** A valid artifact-backed estimate is packaged with the target identity, predictor fingerprint, pipeline version, baseline statistic and limitations. Bound checks are enforced through artifact validation rather than by hiding failures through score clipping.

**Output and purpose.** The structured result can be serialized with `to_dict()` and passed to the policy. The output explicitly states `calibrated=False` and `uncertainty=None`.

**Implementation and failure behavior.** Finite risk and integrity checks prevent invalid artifacts from becoming plausible predictions. Logs record predictor and pipeline identifiers without prompt content by default. The result carries external-validity and no-guarantee warnings.

**Status.** Implemented and tested. This provenance is reproducibility metadata and integrity checking, not a digital signature proving scientific validity or an individualized confidence interval.

## 4.9 Stage 8: Configurable routing

**Inputs and processing.** `Policy.route` receives a prediction and optional explicit verification and escalation thresholds. If thresholds are supplied, they must be finite, bounded and ordered. Using uncalibrated estimates for threshold routing requires explicit opt-in.

**Output and purpose.** The policy returns `allow`, `verify` or `escalate`. This separates estimating a number from deciding what to do with it.

**Implementation and failure behavior.** The default policy escalates because no validated thresholds exist. Missing predictions and invalid risk values also escalate. It does not silently convert an error into permission to generate.

**Status.** Implemented as an action-selection interface. The returned action is not proof that the operational intervention is beneficial. Threshold validation as a software check is different from threshold validation as a scientific decision rule.

## 4.10 Stage 9: Downstream integration

**Inputs and proposed processing.** A future application could receive the selected action and invoke normal generation, retrieval, evidence-grounded answering, independent verification or abstention.

**Output and purpose.** The intended outcome would be a better reliability-and-cost trade-off for delivered answers. Such a claim requires observing the intervention's actual outcomes and overhead.

**Implementation and failure behavior.** The present policy does not execute these actions. Although the repository has an offline generation pipeline, it is not an integrated live routing service. A future application would need explicit error handling and a conservative fallback when any dependency is unavailable.

**Status.** Proposed integration. No retrieval service, live verification executor or measured downstream risk reduction is established by this project.

## 4.11 Stage 10: Offline outcomes, targets and evaluation

**Inputs and processing.** After pre-generation capture, the research pipeline generates ten answers and sends each to the evaluator. Strict aggregation creates one prompt-level target. Offline training joins features and targets by stable identity; evaluation compares held-out predictions with outcomes.

**Output and purpose.** The outputs are preserved response records, evaluation files, strict targets, model comparisons and retrospective routing analyses. These outcomes provide supervision for fitting and scoring, not online features.

**Implementation and failure behavior.** `run_production.py`, versioned evaluator modules, cohort helpers and saved experiment artifacts implement this research path. Matching evaluator versions and complete binary outcomes are required. Failed or inadequate evaluations cannot silently become zero targets.

**Status.** The cohort and analyses are completed. Full nested selection lineage is only partially recovered, so reproducible outcome files do not by themselves certify the entire model-selection procedure.

# Chapter 5: Feature engineering and hidden-state representations

## 5.1 The five prompt features

The final prompt matrix has shape `(486, 5)` in the following order:

| Feature | Meaning | Possible signal and limitation |
|---|---|---|
| `question_chars` | Number of characters in the question | Surface length; not semantic difficulty |
| `context_chars` | Number of characters in the passage | Evidence volume; length alone does not show relevance |
| `question_tokens` | Token count from the frozen tokenizer convention | Model-facing length; dependent on tokenizer |
| `context_tokens` | Token count for the context | Context processing burden; not evidence sufficiency |
| `question_ends_with_question_mark` | Indicator for trailing question-mark structure | Simple formatting cue; likely weak semantic information |

The construction scripts count question and context tokens separately without adding special tokens for these counts. Hidden extraction instead uses the complete rendered chat prompt. These are different objects and their lengths should not be conflated.

No semantic embedding model, entity recognizer or syntactic parser is part of these five features. Earlier configuration documents mention broader categories, but categories in a plan are not implemented measurements. Generated answers, evaluator labels, empirical targets, generation latency, truncation indicators and reference answerability are excluded as predictor inputs.

## 5.2 What the 29 vectors actually represent

For one prompt and one returned layer, the forward pass provides a token-by-hidden-dimension array. The capture function takes a mean across tokens to obtain a 2,048-dimensional vector for that layer. It repeats this for every returned layer, including the embedding output. There are 29 returned entries, corresponding to the embedding representation and 28 decoder-layer outputs under this model's hidden-state convention.

The stored `mean_pooled` array therefore has shape `(29, 2048)` for one prompt. `last_token` has the same shape but uses each layer's last prompt-token state. These are not 29 token vectors. Token pooling has already occurred. The means cover the tokenized rendered prompt, including its instructions and formatting, rather than just the question words.

The final saved hidden matrix has shape `(486, 29, 2048)`: prompts, returned layers, hidden dimensions. A single prompt is not yet represented by only one stored vector. Selecting or aggregating the layer axis is an additional methodological decision.

## 5.3 Two implemented consumption strategies

`extract_hidden_representation` in `src/predictor/baselines.py` implements `last_layer_mean_pooled` by selecting `mean_pooled[-1]`. The older hidden-only and fusion predictor classes use this convention.

In contrast, `run_final_ml3.py`, `cv_run_proper.py` and the expanded benchmark's saved metadata use a mean across axis 1 of the three-dimensional cohort array. This yields `(486, 2048)` by averaging the per-layer token means, including the embedding entry. It is not another average across tokens. It differs from selecting the final decoder layer.

The engineering audit checked three stored cohort rows against their raw hidden JSON and found agreement with `mean_pooled`, not `last_token`. A complete reconciliation of all hidden rows and construction provenance remains a documented follow-up. Some historical construction scripts expect keys not present in the inspected raw schema, so their existence alone does not prove that they reproduce the saved cohort arrays.

Layer averaging avoids having to choose one layer, but mixes representations with potentially different scales and functions. Selecting the last layer is simpler but may discard useful information from earlier layers. Neither choice is proven optimal here. Changing it would define a new experiment, not a harmless reinterpretation of the old results.

## 5.4 Scaling, PCA and fusion

Standard scaling estimates training-feature means and scales so that numeric magnitudes do not dominate a regularized regression merely because of their units. PCA finds directions of variation and reduces dimensionality. With only a few hundred training prompts, directly fitting thousands of hidden dimensions can be unstable. PCA is an engineering response to that imbalance, not a guarantee that the retained variation is relevant to unreliability.

The corrected CV experiment uses `StandardScaler`, then PCA with 50 components for hidden features, fitting both on the training fold. Fusion concatenates the five prompt features with 50 transformed hidden components, then fits the combined regression pipeline inside that fold. A test fold is transformed only using fitted training transformations.

The older standalone classes are not identical: hidden-only uses scaling before PCA and Ridge, while the legacy fusion class fits PCA on training hidden vectors, concatenates prompt features, then scales the combined matrix before Ridge. It also contains numerical substitution with `nan_to_num`. Other legacy helpers impute absent numeric features to zero and tolerate some malformed files. These are limitations to harden before learned deployment; they are not features of the new strict constant interface.

For the expanded benchmark, the report records PCA and reduced-search changes during a resumed run, particularly for tree-based hidden configurations. Per-fold selected estimator parameters were recovered, but full per-fit transformation parameters and the original nested runner were not. I therefore cannot assert that every expanded inner preprocessing boundary has been independently certified.

## 5.5 Why features may not generalize

The prompt features may capture length and formatting shortcuts. Hidden features may capture passage topics, writing style or representation scale rather than transferable risk. All signals depend on the frozen model, tokenizer, rendering and corpus. A small cohort can amplify spurious patterns, while token and layer averaging can dilute localized evidence conflicts.

The experiments show what these particular representations did on the saved cohort. They do not show that hidden states are useless in general, nor that more complex features will solve the task. Any improved representation needs an explicit validation design and a cost justification before further complexity is warranted.

# Chapter 6: Models, baselines and evaluation methodology

## 6.1 Constant baselines and learned models

A mean baseline predicts the training-target average for every test prompt. It is the constant that minimizes training squared error. A median baseline predicts the training-target median, which minimizes training absolute error. In this cohort, the median is zero. These baselines test whether a learned predictor adds value beyond knowledge of the target distribution.

The original fixed-split comparison evaluated Ridge, Random Forest, Extra Trees and histogram-based gradient boosting for prompt-only, hidden-only and fused representations. Ridge used alpha 1.0. Forest specifications used 100 estimators. The corrected CV source uses the same broad fixed-configuration comparison, with 50 hidden PCA components where relevant and random seed 12345.

The expanded saved benchmark contains 39 configurations: six baseline copies and 33 learned configurations, consisting of 11 model families across three feature families. The baseline copies are identical across feature families. They do not constitute six independent prediction methods.

| Model family | Basic idea | Examples of selected parameter values saved in the expanded records |
|---|---|---|
| Ridge | Linear regression with squared coefficient penalty | alpha .01, 1 or 10 |
| Lasso | Linear regression with an absolute coefficient penalty | alpha .0001 or .1 |
| Elastic Net | Combined Ridge and Lasso penalties | alpha .001 or .1; l1_ratio .1, .5 or .9 |
| Support vector regression, SVR | Margin-based nonlinear regression | C .1, epsilon .01, gamma `scale` |
| Nearest-neighbor regression, KNN | Predict from nearby feature rows | 3, 5 or 7 neighbors; uniform or distance weighting |
| Decision tree, DT | Recursive partitions of features | Depth 3, 5, 10 or unrestricted; minimum leaf sizes 1, 3 or 5 |
| Random Forest, RF | Average of randomized decision trees | 50 or 100 estimators; selected depths 5 or 10 |
| Extra Trees, ET | Average of highly randomized trees | 50 or 100 estimators; depths 5, 10 or unrestricted |
| Gradient boosting, GBR | Additive trees fitted to residual structure | Learning rate .01 or .1; 50 or 100 trees; depth 3 or 5 |
| Histogram gradient boosting, HGB | Boosted trees with binned feature splits | Learning rate .01 or .1; 100 or 200 iterations |
| Multilayer perceptron, MLP | Small feed-forward neural regressor | One hidden layer of 50 units; alpha .0001 or .001 |

This table lists values appearing among saved choices, not a recovered full search grid. The expanded records contain five selected-parameter dictionaries for each learned configuration, giving 165 recorded choices. Those dictionaries do not recover all defaults, search trials or selection boundaries. A note on hidden GBR records a reduced search with PCA. No claim is made that every configuration used an identical search budget.

## 6.2 Five distinct evaluation stages

| Stage | Population and design | What it answers | Principal evidence limit |
|---|---|---|---|
| Original fixed split | 340 training prompts and 146 test prompts; seed 12345; fixed model settings | How specified models performed on one split | No separate validation selection in the original comparison; not all row predictions persisted there |
| Corrected training-only CV | Five GroupKFold partitions within the 340 training prompts, then one test on the same 146 prompts | Can selection be confined to training data in that source design? | Inner row predictions were not saved; repeated test reuse prevents a fresh confirmatory claim |
| Expanded benchmark reported as nested CV | Five outer folds over all 486 prompts; 39 configurations | Distribution of saved OOF regression performance | Inner-validation score tables and complete execution lineage are missing |
| Retrospective routing | Fixed threshold grid applied to saved scores on identical 486 prompts | Descriptive risk and coverage trade-offs | No validated joint model and threshold selection; no actual verification outcomes |
| Inference validation | Shipped baseline artifacts, synthetic API cases and local timing workload | Does the supported interface behave correctly and what overhead was observed? | Software correctness is not predictive validity; local timings are not production benchmarks |

The original and corrected fixed-split experiments share a test cohort. They are not independent replications. The expanded benchmark changes the evaluation population and folds. Its results must not be compared as if they were another test of exactly the same fitted model.

## 6.3 Error metrics in plain language

For held-out target y_i and prediction p_i:

```text
MAE  = mean(abs(y_i - p_i))
RMSE = sqrt(mean((y_i - p_i)^2))
R²   = 1 - sum((y_i - p_i)^2) / sum((y_i - mean(y))^2)
```

MAE measures the average absolute error on the risk scale. RMSE gives larger errors more influence because it squares them. R² compares squared error with the constant mean of the evaluated targets. That test-set mean is a reference used to score predictions, not a value supplied to the model. A negative R² means worse squared error than that reference. It does not mean the software failed to execute, and it does not establish performance relative to every other baseline.

If all evaluated targets are equal, the R² denominator is zero and the statistic is undefined. A careful report does not replace that situation with a meaningful-looking score.

A model can reduce MAE while worsening RMSE or R² if it improves many modest errors but makes a few large mistakes. This is relevant to a distribution with many zeros and a smaller set of high risks. No single metric should be chosen after seeing which one looks best.

## 6.4 Cross-validation and selection

Cross-validation fits on one subset and evaluates on held-out rows, repeating across folds. LUX groups by prompt identity so that multiple outcomes for a prompt do not produce train and test examples with the same prompt. In the final risk matrix, each prompt is one row.

Nested CV adds an outer separation. Inner training and validation data choose model settings, while an outer test fold evaluates the resulting procedure. Preprocessing must be included in that boundary. Fitting PCA on all prompts before splitting can leak information about the evaluation distribution even without explicitly using its targets.

The corrected CV source visibly fits scalers and PCA within training folds. For the expanded experiment, stored outer IDs, predictions, fold metrics and selected parameters support partial reconstruction, but do not prove the full inner process. The local search did not find its exact runner or inner prediction tables. This is a missing-evidence limitation; it is not a demonstrated finding that every expanded model leaked labels.

## 6.5 Paired uncertainty estimates

A paired bootstrap resamples prompt identities and compares both methods on the same sampled prompts. This preserves the pairing between a target and each model's prediction. In the corrected fixed-split comparison, 1,000 bootstrap resamples estimate the difference in MAE between the training-mean baseline and the selected model on the 146 test prompts.

The engineering OOF reanalysis uses 2,000 paired prompt resamples. Routing uncertainty uses 2,000 resamples stratified by the saved outer test folds, holding the predictions and action masks fixed. Percentile intervals summarize variation under that resampling scheme. They are not individual prediction intervals, guarantees of correctness, or full uncertainty estimates for rerunning model selection.

These intervals do not account for all shared-context dependence, repeated research decisions or multiple comparisons. Bootstrap cannot repair selection leakage. A recorded fraction of favorable bootstrap differences is also not a posterior probability that the model is truly better.

## 6.6 Why low MAE is not enough for routing

The median predicts zero for every prompt. It can achieve low MAE because most targets are zero, yet it gives no ordering of relative risk. At any positive threshold it allows every prompt; at threshold zero it escalates every prompt. Neither choice selectively finds difficult prompts.

Useful routing therefore needs decision metrics: how many prompts are escalated, how much unreliable outcome mass is captured, what empirical risk remains in the normal path, and what the intervention costs. Calibration is another separate question: whether predicted levels correspond to observed risk levels. A score bounded between zero and one does not establish calibration.

# Chapter 7: Results and interpretation

## 7.1 Dataset and target statistics

| Quantity | Verified value | Interpretation |
|---|---:|---|
| Canonical source scenarios in selection manifest | 130,319 | Input inventory, not the experimental sample size |
| Unique normalized source question-and-context pairs | 130,219 | Exact normalized duplicates were identified |
| Selected prompts | 500 | Preserved existing prompts plus seeded expansion |
| Generated answers | 5,000 | All recorded generation statuses are successful |
| Evaluation files | 5,000 | Final CP4.7 evaluator directory |
| Strict eligible prompts | 486 | One risk target per eligible prompt |
| Excluded prompts | 14 | Every excluded prompt has an inadequate judgment |
| Eligible binary judgments | 4,860 | Ten for each eligible prompt |
| Eligible unreliable judgments | 1,014 | Numerator behind aggregate mean risk |
| Eligible reliable judgments | 3,846 | Remaining eligible judgments |
| Zero-risk prompts | 367 | 75.5144 percent of the eligible cohort |
| Positive-risk prompts | 119 | 24.4856 percent of the eligible cohort |
| Mean target | 0.2086419753 | Eligible unreliable fraction with equal prompt denominators |
| Median target | 0 | Explains the zero median baseline |
| Population standard deviation | 0.3867765481 | Computed with the population convention used in verification |
| Minimum and maximum target | 0 and 1 | Empirical range, not uncertainty bounds |

Sources are the expanded selection and generation artifacts, the final strict snapshot, `y.npy`, and the nested verification report. The population standard deviation is not a standard error of the mean.

## 7.2 Original fixed-split results

The table below is transcribed from `experiments/final_cohort_2026_10_24/model_comparison.json`. It reports the one 340-training and 146-test split. It must not be described as the expanded OOF benchmark.

| Fixed-split configuration | Test MAE | Test RMSE | Test R? |
|---|---:|---:|---:|
| baseline | 0.314021 | 0.383036 | -0.001978 |
| ridge_prompt | 0.315697 | 0.384700 | -0.010700 |
| ridge_hidden | 0.315337 | 0.394089 | -0.060638 |
| ridge_fusion | 0.313035 | 0.393379 | -0.056817 |
| rf_prompt | 0.297240 | 0.382846 | -0.000983 |
| rf_hidden | 0.274877 | 0.401278 | -0.099688 |
| rf_fusion | 0.272781 | 0.398045 | -0.082036 |
| et_prompt | 0.303767 | 0.390348 | -0.040598 |
| et_hidden | 0.281692 | 0.391414 | -0.046290 |
| et_fusion | 0.286089 | 0.394435 | -0.062502 |
| hgb_prompt | 0.317597 | 0.414066 | -0.170893 |
| hgb_hidden | 0.297143 | 0.414245 | -0.171904 |
| hgb_fusion | 0.299358 | 0.406937 | -0.130922 |

The fixed-split fusion RF has lower recorded MAE than the mean baseline, but its R² is negative. The later corrected CV selects a hidden RF through training-only CV rather than selecting the lowest test error from this table. That distinction protects the intended selection boundary. The original result file does not preserve all per-model test prediction rows, so its complete metrics are available as recorded values rather than all being independently reproduced from persisted row predictions.

## 7.3 Corrected training-only CV results

| Measure | Verified result |
|---|---:|
| Training prompts | 340 |
| Test prompts | 146 |
| Training CV folds | 5 |
| Selected configuration | RF with hidden features |
| Selected configuration's mean inner MAE | 0.273064706 |
| Standard deviation of its five inner MAEs | 0.031859825 |
| Test MAE, mean baseline | 0.314020951 |
| Test MAE, selected RF | 0.274876712 |
| Test RMSE, mean baseline | 0.383036196 |
| Test RMSE, selected RF | 0.401278180 |
| Test R², mean baseline | -0.001977984 |
| Test R², selected RF | -0.099688387 |
| Paired MAE gain, baseline minus model | 0.039144239 |
| Relative MAE gain against the mean baseline | 12.47 percent |
| Bootstrap 95 percent interval for MAE gain | [0.016852790, 0.059941448] |

The five validation MAEs were 0.291618, 0.234279, 0.324926, 0.252956 and 0.261544. The saved 146 test predictions and IDs allow row alignment and test metric reproduction. Source inspection supports confinement of the CV comparison to the training split.

The paired interval favors this selected model over the mean baseline for MAE on that fixed test set. It does not establish superiority to the median baseline on the full cohort. Its RMSE and R² are worse than the mean baseline's recorded test values. Earlier narrative summaries use inconsistent language about statistical conclusiveness; the numerical interval is reported here with its limited conditional interpretation rather than converted into a broad success claim.

## 7.4 Expanded OOF benchmark

The following table includes both distinct baselines and every learned configuration. Four redundant baseline copies from hidden and fusion families are omitted because their predictions are identical. Values come from `nested_cv_results.json`; the verification artifact independently recomputes the pooled metrics from OOF rows.

| Expanded configuration | OOF MAE | OOF RMSE | OOF R? |
|---|---:|---:|---:|
| dummy_mean (prompt) | 0.317064 | 0.386904 | -0.000659 |
| dummy_median (prompt) | 0.208642 | 0.439463 | -0.290993 |
| ridge (prompt) | 0.317543 | 0.388087 | -0.006788 |
| lasso (prompt) | 0.318938 | 0.388959 | -0.011315 |
| elasticnet (prompt) | 0.318930 | 0.388945 | -0.011245 |
| svr (prompt) | 0.214182 | 0.434953 | -0.264634 |
| knn (prompt) | 0.306470 | 0.431370 | -0.243884 |
| dt (prompt) | 0.333210 | 0.454427 | -0.380409 |
| rf (prompt) | 0.324719 | 0.406537 | -0.104793 |
| et (prompt) | 0.319657 | 0.395347 | -0.044810 |
| gbr (prompt) | 0.324823 | 0.401877 | -0.079608 |
| hgb (prompt) | 0.321112 | 0.398047 | -0.059127 |
| mlp (prompt) | 0.314222 | 0.400365 | -0.071498 |
| ridge (hidden) | 0.394854 | 0.497664 | -0.655590 |
| lasso (hidden) | 0.317064 | 0.386904 | -0.000659 |
| elasticnet (hidden) | 0.317521 | 0.387765 | -0.005119 |
| svr (hidden) | 0.223997 | 0.422184 | -0.191469 |
| knn (hidden) | 0.310406 | 0.424087 | -0.202233 |
| dt (hidden) | 0.311680 | 0.474396 | -0.504392 |
| rf (hidden) | 0.303474 | 0.383895 | 0.014843 |
| et (hidden) | 0.310578 | 0.384885 | 0.009756 |
| gbr (hidden) | 0.300624 | 0.392408 | -0.029333 |
| hgb (hidden) | 0.305694 | 0.396638 | -0.051645 |
| mlp (hidden) | 0.717316 | 0.897665 | -4.386518 |
| ridge (fusion) | 0.396193 | 0.498794 | -0.663114 |
| lasso (fusion) | 0.317064 | 0.386904 | -0.000659 |
| elasticnet (fusion) | 0.317521 | 0.387765 | -0.005119 |
| svr (fusion) | 0.224017 | 0.422202 | -0.191574 |
| knn (fusion) | 0.305968 | 0.418881 | -0.172900 |
| dt (fusion) | 0.296684 | 0.464977 | -0.445247 |
| rf (fusion) | 0.304071 | 0.384873 | 0.009818 |
| et (fusion) | 0.309409 | 0.383472 | 0.017014 |
| gbr (fusion) | 0.318777 | 0.410049 | -0.123960 |
| hgb (fusion) | 0.310777 | 0.394537 | -0.040530 |
| mlp (fusion) | 0.865704 | 1.122945 | -7.429400 |

No learned configuration has lower pooled OOF MAE than the median baseline. The closest is prompt SVR, whose MAE is 0.214182 compared with 0.208642. The paired engineering analysis reports a median-minus-SVR MAE gain of -0.005539654, with a conditional 95 percent interval [-0.006354844, -0.004722160]. The negative sign means the SVR has higher absolute error than the median under this comparison. The relative gain is approximately -2.66 percent.

The mean baseline is better suited to squared error than the median: its RMSE is 0.386904 compared with 0.439463. Hidden RF reaches RMSE 0.383895 and R² 0.014843, while fused ET reaches RMSE 0.383472 and R² 0.017014. These are modest improvements on those metrics. Their MAEs, 0.303474 and 0.309409, remain substantially higher than the median's. The results do not justify hiding the unfavorable MAE comparison or declaring one metric universally correct.

### Fold distributions

| Configuration | Fold 0 MAE | Fold 1 MAE | Fold 2 MAE | Fold 3 MAE | Fold 4 MAE |
|---|---:|---:|---:|---:|---:|
| Mean | 0.310856 | 0.305430 | 0.316678 | 0.328423 | 0.323998 |
| Median | 0.195918 | 0.195876 | 0.204124 | 0.229897 | 0.217526 |
| Prompt SVR | 0.201876 | 0.200961 | 0.210414 | 0.235176 | 0.222607 |
| Hidden RF | 0.294819 | 0.298766 | 0.291628 | 0.321949 | 0.310295 |
| Fused ET | 0.298420 | 0.299897 | 0.308199 | 0.318559 | 0.322083 |

Fold 0 contains 98 test prompts and the others contain 97 each. Aggregate MAE weights prompts, not an arbitrary unweighted average of unequal fold sizes. The median beats the prompt SVR on all five displayed fold MAEs. All 195 fold MAEs across the 39 saved configurations were reconciled in the recovery audit. This supports numerical alignment, not certification of every historical inner fit.

### What the recovered lineage adds

The benchmark contains 165 selected-parameter entries, not a complete absence of model-selection metadata. However, no inner-validation prediction table, complete inner split manifest or per-fold threshold-selection result was found. The checkpoint duplicates the final result rows. `nested_cv_best.json` contains the minimum global OOF-MAE row, which is a ranking across outer outcomes rather than a valid per-outer-fold policy selection.

Thus the benchmark can be reported as saved OOF evidence with verified alignment and partial lineage. It cannot be used to claim that a final model and threshold were selected without evaluation-fold influence.

## 7.5 Retrospective routing results

The routing analysis examined every declared threshold for each risk-bound-valid saved configuration. Thirteen of the 33 learned configurations had at least one prediction outside the valid risk range and were excluded as whole configurations. Scores were not clipped, and prompts were not selectively dropped. This left 20 learned configurations and two distinct baselines. Eight thresholds produced 176 evaluated policies.

The grid was 0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75 and 1. A score at or above the threshold was routed to verification in the simulation. These values are sensitivity cases, not validated deployment thresholds. All policies used the same 486 prompts and original five outer folds.

The next table shows one declared grid point, 0.2, together with the no-estimator reference. It is an illustration, not a chosen threshold or winning-model table. The complete 176-policy curves, per-fold metrics and paired comparisons are preserved in the routing result file and report.

| Policy at threshold 0.2, unless stated | Normal prompts | Escalation rate | Normal-path empirical risk | Captured unreliable mass |
|---|---:|---:|---:|---:|
| Normal generation, no estimator | 486 | 0.00 percent | 0.208642 | 0.00 percent |
| Median baseline | 486 | 0.00 percent | 0.208642 | 0.00 percent |
| Mean baseline | 0 | 100.00 percent | Undefined | 100.00 percent |
| Prompt SVR | 486 | 0.00 percent | 0.208642 | 0.00 percent |
| Hidden RF | 287 | 40.95 percent | 0.172474 | 51.18 percent |
| Fused ET | 263 | 45.88 percent | 0.169582 | 56.02 percent |

The normal-path mean target is also the analysis's generation-level false-reassurance proxy. Among normal-path prompts, the fraction with any unreliable attempt is a separate quantity: 24.49 percent without routing, 20.56 percent for the illustrated hidden RF, and 19.77 percent for the illustrated fused ET. Those prompt fractions must not be confused with the mean empirical risk in the table.

Non-predictive hash routing uses a fixed salt and prompt IDs without inspecting outcomes. One reference with a nominal 25 percent budget escalates 120 prompts, or 24.69 percent after per-fold rounding, and has normal-path risk 0.204098. A nominal 50 percent budget escalates 241 prompts, or 49.59 percent, with normal-path risk 0.220816. Variation in one fixed hash realization is not evidence that random routing intrinsically increases or decreases risk.

Count-matched hash references provide a fairer burden comparison for each candidate. At threshold 0.2, hidden RF minus its count-matched hash reference has a normal-risk difference of -0.039721, with paired 95 percent interval [-0.086435, 0.002953]. Fused ET has difference -0.039924 and interval [-0.085312, 0.003954]. Both intervals include zero. These are conditional, unadjusted comparisons among many descriptive cases, not selection results or deployment guarantees.

The analysis also reports remaining unreliability under hypothetical verification. If escalated prompts retain fraction q of their original empirical risk, the workload-normalized quantity is:

```text
remaining = (sum(y_i for allowed prompts)
             + q * sum(y_i for escalated prompts)) / all original prompts
```

For the illustrated hidden RF, this is 0.101852 at q=0, 0.155247 at q=0.5 and 0.208642 at q=1. The first assumes perfect removal, the second assumes halving and the last assumes no benefit. None was measured as an intervention. If the action is abstention instead, the honest outputs are reduced coverage and the empirical risk among delivered answers, not a claim that abstained answers became correct.

The policy-selection audit explains why the available OOF table cannot safely serve as both threshold-selection data and final testing. A row in another outer fold may have a score produced using the current test fold's outcomes during fitting or selection. Filtering the current fold's rows from the table does not remove that dependence. No model or threshold was selected as a validated operational policy.

## 7.6 Engineering verification results

| Verification stage | Recorded outcome | Scope of the evidence |
|---|---|---|
| Initial inference and existing predictor checks | 47 passed | Engineering implementation stage |
| Initial broader suite | 355 passed | Earlier code snapshot, not an additional independent research experiment |
| Final initial-inference focused checks | 27 passed | Includes provenance-bound exporter checks |
| End-to-end focused checks | 75 passed in 15.97 seconds | Public inference, artifacts, failure paths, workload cases and related analysis helpers |
| End-to-end broader suite | 409 passed in 273.07 seconds | Suite collection snapshot; two later test additions were in the final focused run |
| Routing and relevant existing tests | 91 passed in 9.18 seconds | Includes 16 routing tests and relevant inference and OOF tests |
| Lineage recovery | 165 parameter entries and 195 fold MAEs checked | Read-only artifact audit, not retraining |

Test counts are historical command outcomes documented in the engineering reports. They overlap and must not be added into a single total. A durable full raw log for every earlier command was not recovered as part of this report, so those counts are identified as recorded verification outcomes, not newly executed test results.

## 7.7 Local operational measurements

| Quantity | Saved local result |
|---|---:|
| Fresh-process samples | 20 |
| Warm workload cases | 12 |
| Warm measured requests | 12,000 |
| Cold process median | 4.266744 seconds |
| Cold process mean | 6.138483 seconds |
| Cold process 95th percentile | 16.010238 seconds |
| Module-import median | 3.808323 seconds |
| Artifact-load median | 0.692 milliseconds |
| First prediction and routing median | 0.2089 milliseconds |
| Range of warm case means | 0.089948 to 0.109743 milliseconds |
| Final peak working set | 122,593,280 bytes, approximately 116.9 MiB |

The host was Windows build 10.0.26200 with eight logical Intel64 processors and about 15.79 GiB physical RAM. The warm workload used six synthetic text lengths from 32 to 65,536 characters, each as text and as structured input with a Unicode question. Each case had 1,000 measured sequential requests after warmup.

Timing used `perf_counter_ns`; memory used native Windows process working-set counters. Warm timings included prediction and routing but excluded serialization and assertions. Cold process timing included interpreter startup, import, artifact loading, the first request and process exit. Filesystem caches were not cleared. Regression tests overlapped the probe, so contention is a limitation.

The eager imports in `src/predictor/__init__.py` load legacy predictor dependencies even when importing the constant interface. The saved import timing explains most of the observed cold-start cost. This is not Qwen loading time. No target model was loaded in this probe. These numbers are local microbenchmarks, not production latency, throughput or memory guarantees.

# Chapter 8: Implementation and software engineering

## 8.1 What is supported now

The supported entry point is:

```python
predict_risk(prompt, target_model, model_revision, config, *, artifact)
```

It is a Python function, not a deployed network service. It supports the single frozen target configuration and explicitly loaded constant artifacts. The default example uses the cohort mean, not a learned winner from one experiment. The mean is chosen for its expected-risk interpretation, while the median is separately exposed for absolute-error comparison. Neither can rank accepted prompts.

The following example describes the existing interface. It was not executed as a new experiment for this report:

```python
from src.predictor.inference import (
    Artifact, GENERATION_CONFIG, MODEL, REVISION, Policy, predict_risk,
)

artifact = Artifact.load("docs/engineering/baseline_mean_v2.json")
prediction = predict_risk(
    {"question": "What does the passage establish?", "context": "A passage."},
    MODEL,
    REVISION,
    dict(GENERATION_CONFIG),
    artifact=artifact,
)
result = prediction.to_dict()
action = Policy().route(prediction)  # conservative default: escalate
```

The mean output is approximately 0.208642 for every valid prompt. The median output is zero for every valid prompt. A value of zero from the median is a property of that estimator and target distribution, not permission to trust an answer.

## 8.2 Artifact contract

The v2 baseline files use artifact schema version 1. The file suffix identifies the engineering export revision; it is not a claim that the schema version is 2. Both artifacts use feature pipeline `constant-no-features-v1`.

| Artifact field | Role |
|---|---|
| `schema_version` | Selects the accepted serialization contract |
| `feature_pipeline_version` | Identifies the empty-feature constant pipeline |
| `target_model`, `model_revision` | Bind the prediction to the frozen target |
| `generation_config` | Records the supported decoding and execution settings |
| `statistic`, `value`, `n_train` | State the baseline definition and fitted cohort size |
| `training_ids_sha256` | Fingerprints the training-ID set |
| `source_sha256` | Records hashes of the target snapshot and generation artifact |
| `predictor_id` | Fingerprints the artifact body |

The exporter checks the audited source hashes, strict integer counts, excluded-outcome counts and expected eligible size. It rejects changed source snapshots rather than silently branding them with the frozen configuration. Saving uses exclusive creation so an existing version is not overwritten.

The baseline values are fitted on all eligible targets for use as cohort references. That export is not a new held-out performance experiment. The regression tables use the appropriate saved held-out or fold-trained baselines, not an all-data mean supplied to a test fold.

## 8.3 Output contract and error handling

The output fields are `estimated_risk`, `predictor_id`, `target_model`, `model_revision`, `feature_pipeline_version`, `baseline_statistic`, `calibrated`, `uncertainty` and `warnings`. Provenance makes the estimate traceable. The warnings state that the result is a cohort baseline, external validity is unverified and no correctness guarantee is provided.

Missing files raise file errors. Invalid inputs, incompatible schemas, malformed artifacts and configuration mismatches raise explicit exceptions. The caller may catch those failures and pass no prediction to the conservative policy. No path automatically returns a low-risk value because feature extraction or model loading failed.

Fallback is explicit. An application can try a preferred artifact and deliberately load a known baseline if appropriate. The selected baseline remains observable through its statistic, pipeline version and fingerprint. The implementation does not silently replace a learned score with zero. It also does not claim that the learned artifact loader is a production-supported component of this interface.

## 8.4 Routing contract

`Policy` accepts `verify_at`, `escalate_at` and `allow_uncalibrated`. With no thresholds, it escalates. Explicit thresholds must satisfy the software's ordering and range checks. If a caller elects to use the uncalibrated baseline for routing, it must opt in. A missing prediction or invalid risk value escalates.

The software can enforce a threshold contract without validating the threshold's utility. Choosing operational thresholds requires validation data and a declared cost of errors, verification, abstention, latency and compute. Since a constant treats all prompts alike, baseline-only routing can change global workflow policy but cannot allocate intervention selectively by prompt risk.

## 8.5 Leakage boundaries, logging and reproducibility

The online prompt mapping is an allowlist, not a permissive generation-record parser. Extra fields such as generated answers, evaluator labels, hidden references or outcome metrics are rejected. The constant implementation computes no prompt features, so even accepted text is not transformed into a score-dependent representation. This does not semantically classify all text a caller might include; it guarantees the implemented constant does not consume it as a learned feature vector.

Structured logs contain predictor and pipeline identifiers without logging prompt content by default. This reduces accidental disclosure, though an application's own logging still needs care. Credentials belong in environment variables; no credentials are needed for constant inference.

Model revision pinning, source hashes, stable IDs and saved folds support reproducibility. The project is not fully environment-locked: dependency ranges and missing historical runner details remain obstacles to exact reconstruction of every experiment. The present strict artifact layer is stronger than some older training helpers, which still contain zero imputation, first-match joins and malformed-file skipping. Those old behaviors were documented rather than concealed by a large rewrite.

Tests cover output shape and bounds, artifact round trips and tampering, model and configuration incompatibility, explicit fallback, invalid prompts, prohibited fields, deterministic repeated requests, log content, policy failures and OOF alignment. Passing those tests establishes software behavior within the tested contract. It does not establish calibration, real-world predictive skill or an effective intervention.

# Chapter 9: Limitations and unresolved issues

## 9.1 Sample size and population

Four hundred and eighty-six prompts are small relative to the diversity of real applications, especially for high-dimensional hidden states. The source is one passage-grounded dataset. Retained earlier prompts, exact-content deduplication and strict exclusion create a particular cohort rather than a general distribution of user queries. Shared context or topic can connect otherwise distinct prompt IDs, so disjoint IDs do not establish independence at every level.

Only one target model, revision and configuration has been evaluated. No result establishes transfer to a different model size, architecture, tokenizer, decoding setting or prompt template. Broader claims need separate targets and labels under those conditions.

## 9.2 Proxy outcomes and repeated generations

Evaluator judgments may be biased or inconsistent. The definition of materiality is qualitative and tied to the supplied passage. Ten generations give a coarse and potentially correlated risk estimate. Zero observed failures do not prove zero underlying risk. Exclusion of inadequate prompts may remove cases that matter operationally.

The final targets are therefore useful measurements for this study, not independently verified universal truth labels. Their limitations carry through the regressor and routing analyses.

## 9.3 Statistical and model-selection limitations

The zero-heavy distribution makes median MAE a demanding and potentially misleading single-number target. The learned models do not establish superiority over it on OOF MAE. Small improvements in RMSE or R² do not establish practical ranking, and negative results do not prove that every pre-generation representation is uninformative.

The 165 recovered hyperparameter dictionaries provide partial lineage. The missing inner validation predictions, exact inner assignments, complete preprocessing histories and original expanded runner prevent certification of a joint model and threshold selection. Global OOF ranking cannot replace that evidence. Earlier fixed-split comparisons and repeatedly inspected cohorts are not pristine new tests.

## 9.4 Uncertainty, calibration and thresholds

There is no demonstrated calibration of individualized predictions and no reliable individual uncertainty interval. Bootstrap intervals concern aggregate evaluation under explicit resampling assumptions. Risk bounds and interval formatting do not create calibration or guarantees.

No deployment threshold has been scientifically validated. The policy API permits thresholds as configuration, while the retrospective grid explores sensitivity. Neither operation estimates a trustworthy threshold for an unspecified real-world cost function.

## 9.5 Engineering readiness versus scientific validity

The constant interface has tested failure handling and reproducible artifacts. It is operationally inspectable, but not an individualized hallucination-risk detector. It does not execute retrieval or verification. Learned deployment, out-of-domain detection, model-loading overhead for a learned hidden path and sustained production capacity have not been validated.

The local warm timings are small because constant inference performs no Qwen forward pass. Comparing them with a hypothetical hidden-state predictor would be misleading. The cold import overhead and legacy training-code weaknesses also remain real engineering limitations.

Most importantly, no experiment has shown that using LUX reduces unreliable answers delivered by an application. A working function and a hypothetical perfect-verification calculation are not substitutes for downstream evidence.

# Chapter 10: Contributions and novelty

## 10.1 Implemented engineering contributions

The project provides a passage-grounded research data pipeline with stable prompt identity, frozen target metadata, saved generation outcomes, versioned evaluator records and strict target aggregation. It implements prompt-only, hidden-only and fusion research predictors, together with baseline comparisons.

The later engineering work adds a narrow inference contract, source-bound JSON baseline artifacts, explicit input and version validation, conservative action selection, observable fallback, prompt-free logging defaults and critical boundary tests. These contributions make the project's current claims easier to inspect and reproduce.

## 10.2 Completed empirical contributions

The completed 500-prompt cohort and 486-prompt strict target dataset support comparisons across several regression families. The expanded OOF verification exposes the strength of the median baseline, metric-dependent model behavior and weak evidence for dependable learned improvement. The routing analysis makes the distinction between low regression error and useful selective decisions concrete.

These are within-project empirical contributions. The repository does not provide the external benchmark comparisons or literature evidence needed to claim state-of-the-art performance or a globally novel model architecture.

## 10.3 Methodological observations

Three observations are especially useful for explaining this project. First, a zero median can win an error metric without ranking any prompt. Second, the layer axis of a stored hidden array cannot be interpreted from its shape alone; construction and consumption must be traced. Third, saved OOF scores do not automatically provide the validation lineage needed for a second layer of model and policy selection.

The recovery audit also corrected an overly narrow schema view by identifying 165 saved parameter choices while retaining the distinction between recorded choices and verified selection evidence. Preserving that nuance is more defensible than describing either complete reproducibility or a complete absence of metadata.

## 10.4 What is still a hypothesis

The idea that richer pre-generation model signals can support useful risk-sensitive decisions remains worth studying. It has not been established by the current results. Fusion is not demonstrated to be consistently best, hidden states are not proven calibrated confidence signals, and routing has not been shown to improve live reliability.

I regard the useful contribution as an experimentally grounded investigation with explicit failure modes and a reproducible negative result. It shows what evidence is missing before this kind of system should be trusted, rather than turning a plausible motivation into an unsupported success claim.

# Chapter 11: Future work

All work in this chapter is proposed. The review-preparation task did not authorize or execute it.

## 11.1 First priority: a leakage-safe decision experiment

Reuse the 486 strict targets and existing five outer folds. Freeze the operational action semantics and costs first. A conservative primary experiment can compare normal answering with escalation that withholds an unverified answer. Its inner-validation objective can combine empirical unreliable-answer cost with escalation or abstention cost. Actual verification that returns a repaired answer requires an additional measured or explicitly assumed effectiveness model.

For each outer fold, create validation predictions using only that outer training partition. Fit all learned preprocessing inside its inner training subsets. Select a small candidate model and threshold pair from the resulting validation outcomes, freeze the pair, refit only on the outer training partition and evaluate once on its test partition. Aggregate those five selected procedures without retrospectively choosing a global winner from their outer results.

The recovery audit proposes at most 20 learned fits for one fixed prompt-only candidate with three inner folds and selected outer refits. A single inner holdout could reduce this to at most 10 fits but offers less validation support. Including prompt, hidden and fusion candidates increases the bound to 50 fits. Those are proposed computation counts, not executed experiments or runtime estimates. No new target-model generations or evaluator calls are needed for that retrospective protocol.

Every score should carry its fit ID, outer and inner membership, source hashes and complete settings. Validation objective tables, costs, threshold grids, tie-breaks and failed candidates should be saved, not just the winner. Existing historical outputs must remain unchanged.

## 11.2 Improve signals only after the evaluation design is fixed

Before adding complexity, reconcile feature schemas and hidden construction for the complete cohort. Then investigate whether a justified alternative layer representation, evidence-compatibility feature or other pre-generation signal adds stable value under the same outer boundary. Such ideas should be assessed by their cost and held-out contribution, not by an appealing interpretation of a hidden vector.

Full target-model loading and extraction latency must be measured for any proposed hidden deployment path. The existing constant timing results cannot stand in for those measurements.

## 11.3 Calibration and individual uncertainty

Start with no calibration in the minimum decision experiment. If a calibrator is later justified, its fitting data must remain separate from the validation data used to compare it and from the outer test fold. A transparent option is separate base-fit, calibration and policy-selection subsets inside each outer training partition. A more data-efficient cross-fitting design needs equally explicit lineage.

Evaluate calibration with held-out predictions and adequate bin support, alongside predictive error. Consider uncertainty methods only with clear assumptions and coverage checks. No method should be presented as a guarantee about the next generated answer.

## 11.4 Generalization and independent confirmation

Extend evaluation to different prompt sources, lengths, question types and answerability groups when valid metadata and labels are available. Study context-grouped splits to distinguish prompt-level interpolation from broader topic generalization. Evaluate other target models and configurations separately rather than transferring the frozen model's score without evidence.

Because the current data have been repeatedly inspected, a genuinely untouched cohort is needed for stronger confirmatory deployment claims. Optional independent label validation could estimate evaluator bias without making it a prerequisite for the current interface.

## 11.5 Downstream and operational evaluation

Compare normal generation without LUX, LUX-informed intervention, and simple non-predictive routing at matched intervention budgets. Measure actual unreliable-answer rates, false reassurance, coverage, verification and escalation rates, latency, computation and group performance. Keep retrieval and verification failures visible.

An intended deployment environment also needs isolated startup measurements, sustained and concurrent requests, memory tracking and a defined latency budget. Reducing eager legacy imports may improve baseline startup, but that is a future engineering change, not a demonstrated improvement in this report.

# Chapter 12: Final conclusion

LUX established a controlled way to study pre-generation signals against empirical unreliability outcomes for one frozen language model. It completed a 500-prompt generation and evaluation cohort, constructed 486 strict risk targets, compared feature families and regression models, audited saved predictions, and built a tested baseline-only inference interface.

The evidence also limits the conclusion. Learned models did not beat the median baseline on expanded OOF MAE. Some metrics improved for some models, but individualized calibration, validated threshold selection and downstream reliability benefits remain unproven. Partial hyperparameter lineage was recovered; the missing inner-validation evidence still prevents a validated joint selection claim.

For real-world use, LUX would need a fully recorded selection procedure, an independently evaluated decision rule, justified calibration or uncertainty claims where needed, and a downstream comparison that measures both reliability and cost. Until then, the appropriate outcome is a transparent research prototype with robust baseline behavior and clear scientific limits.

# Chapter 13: Five-minute presentation script

The following script is intended for approximately five minutes at a measured speaking pace. The architecture diagram can remain visible while explaining the two paths.

Good morning. My project is LUX, which stands for Language Uncertainty Extractor. I investigated whether we can estimate the risk of an unreliable language-model answer before the answer is generated.

The motivation is practical. If a system could identify a difficult prompt in advance, it might retrieve better evidence, reserve verification for that prompt, or avoid giving an unsupported answer. But that only helps if the prediction is better than a simple baseline and the additional work is worth its cost. My project does not claim that hallucinations have been prevented.

I used a frozen Qwen3 model with 1.7 billion parameters. I kept its revision and generation settings fixed so that the outcome being predicted had a consistent definition. The prompts came from SQuAD version 2, which provides questions and passages, including questions that the passage cannot answer.

For each of 500 prompts, the model generated ten answers using recorded sampling seeds. A separate evaluator judged the answers against the passage. It used three labels: reliable, unreliable and inadequate. I retained a prompt only if all ten evaluations were successful and gave a reliable or unreliable judgment. Fourteen prompts contained inadequate judgments, leaving 486 eligible prompts.

The target is simple: the number of unreliable answers divided by ten. It is an empirical rate based on evaluator judgments. It is not a guarantee about the next answer, and the evaluator itself can be wrong.

There are two paths in the architecture. The offline research path extracts features before generation, then generates and evaluates answers to create the targets used for training. The online path must not receive those outcomes. In the currently supported interface, that online path uses an explicit constant baseline artifact and returns its estimate with provenance and warnings.

For research, I compared five surface features, such as question and passage length, with the target model's hidden representations. An important detail is that each stored prompt has 29 layer vectors, each already averaged over prompt tokens. The saved benchmark averages the layer axis, while an older helper selects the last layer. Those are different representations.

I compared prompt-only, hidden-only and fused predictors with mean and median baselines. The strongest caution came from the target distribution: about three quarters of prompts had zero observed risk. This makes the zero median difficult to beat on mean absolute error, even though it cannot rank prompts at all.

In the expanded out-of-fold results, the median MAE was about 0.209. The lowest learned MAE was about 0.214, so the learned models did not beat it. Some hidden and fused models improved RMSE or R-squared slightly, but that does not establish dependable real-world prediction.

I also examined 176 fixed-threshold routing policies. Some showed retrospective trade-offs between escalation and remaining empirical risk. However, I did not select a winning policy. Although 165 per-fold parameter choices were recovered, the inner-validation predictions and full selection lineage were missing. Using the same outcomes to choose and assess a threshold would make the result optimistic.

The engineering contribution is a tested baseline interface with strict configuration checks, versioned artifacts, explicit fallback and conservative routing. The final focused validation recorded 75 passing tests. Those tests establish behavior, not hallucination-detection accuracy.

My main contribution is therefore an inspectable empirical investigation, including a negative result and a clear account of what would be required next. The next research step is a small, properly nested policy-selection experiment, followed by an actual downstream reliability-and-cost comparison. I have designed that work but have not presented it as completed.

# Chapter 14: Examiner questions and model answers

## Q1. Why try to predict risk before generation?

**Answer.** A useful estimate could influence the workflow before generation cost is paid or an unsupported answer is exposed. It could help allocate retrieval or verification. Unlike a post-generation checker, it cannot inspect the actual answer, so the evidence burden is higher. LUX investigates that possibility; it has not demonstrated the operational benefit.

## Q2. Why use Qwen3-1.7B?

**Answer.** It is an open model whose prompt-conditioned hidden states can be inspected in the local research environment. Its size made the CPU-based protocol feasible. The checkpoint and revision were fixed to define a consistent target. The project does not establish that this model is optimal or representative of every model family.

## Q3. Why use SQuAD v2.0?

**Answer.** It supplies a question, a passage and answerability structure, allowing a controlled passage-grounded task. Unanswerable questions make abstention relevant. Its limitation is domain concentration: passage answering is narrower than open-world factual assistance, so performance on this dataset cannot be assumed to transfer to general user prompts.

## Q4. Why generate ten answers per prompt?

**Answer.** Repeated generations support an empirical prompt-level rate rather than one binary observation. Ten was the locked practical protocol under generation and evaluator constraints. It provides 0.1 resolution, not a statistically optimal sample size or a precise latent probability. Correlation between attempts can reduce the effective information further.

## Q5. How is empirical risk calculated?

**Answer.** For a strictly eligible prompt, it is the count labeled `UNRELIABLE` divided by ten. For example, three unreliable judgments and seven reliable judgments give 0.3. This example explains the formula; it is not a new measured result. The target is absent if the strict eligibility rule is not met.

## Q6. Why exclude inadequate evaluations?

**Answer.** An inadequate response is not automatically established as materially unreliable, and it should not silently become reliable either. Requiring ten binary judgments keeps the denominator and target interpretation consistent. The cost is selection bias: ambiguous prompts may be excluded, so the eligible cohort is not equivalent to all selected prompts.

## Q7. Why are the evaluator labels only proxies?

**Answer.** Another model produces them using passage-grounded instructions and a qualitative materiality judgment. It can misunderstand evidence, be inconsistent or share biases across answers. The saved labels are reproducible research observations, not independently established truth. Predicting them well would still require further work before claiming general factual correctness.

## Q8. What exactly is available before generation?

**Answer.** The prompt and supplied context, target identity and configuration, tokenizer-derived lengths, and hidden states from a prompt-only forward pass are available. The future answer, evaluator decision, generated-token count, answer latency and empirical risk are not. Reference answerability is also excluded because it is benchmark information, not an observed prompt feature.

## Q9. How do you extract hidden states without generating an answer?

**Answer.** The code tokenizes the rendered prompt and calls the frozen model with `output_hidden_states=True` under `torch.no_grad()`. This is a forward pass over the input sequence, performed before `generate`. It saves every returned layer's last prompt-token vector and mean across prompt tokens. It does not inspect generated answer tokens.

## Q10. Why are there 29 layer vectors?

**Answer.** The returned tuple includes the embedding output and the decoder-layer outputs. Under this Qwen checkpoint's recorded convention, that is 29 entries, each with 2,048 hidden dimensions after token pooling. The layer axis is not a token axis, and it should not be described as 29 sampled generations.

## Q11. Is each prompt stored as one hidden vector?

**Answer.** Not in the cohort array. It is stored as 29 per-layer vectors, giving `(486, 29, 2048)` for the cohort. The last-layer helper selects one of those vectors. The later comparison averages the layer axis to create one 2,048-dimensional vector per prompt. Those are distinct downstream choices.

## Q12. Why use scaling and PCA?

**Answer.** Scaling prevents units from dominating certain regressors. PCA reduces a high-dimensional representation to a smaller set of directions, which can make learning more manageable with few prompts. Both learn from data and must be fitted only on training rows. PCA preserves variation, not necessarily the variation relevant to unreliability.

## Q13. Why compare prompt-only, hidden-only and fused features?

**Answer.** This separates the possible contributions of surface structure and model-conditioned information. Fusion tests whether the sources are complementary. If only fusion were evaluated, an improvement could not be attributed to either source. The saved results do not establish that fusion consistently provides useful additional signal.

## Q14. What is the median baseline?

**Answer.** It predicts the median of the training risks for every evaluated prompt. The training medians in the saved outer folds are zero. It minimizes training absolute error among constants, but does not use the prompt to distinguish risk. It is a necessary comparator, not an individualized safety model.

## Q15. Why does the median perform well?

**Answer.** About 75.5 percent of eligible targets are zero. Predicting zero makes no error on those rows, which strongly benefits MAE. This illustrates why target distribution matters. Its MAE can be competitive even while its RMSE is worse and its routing gives no useful relative ordering of prompts.

## Q16. What do MAE, RMSE and R² tell you?

**Answer.** MAE is average absolute risk error. RMSE penalizes large errors more strongly. R² compares squared error with predicting the evaluated target mean. Negative R² indicates worse squared error than that reference. They describe regression quality from different perspectives; none directly measures whether a routing intervention improves delivered answers.

## Q17. Why can MAE improve while R² remains negative?

**Answer.** A model can reduce many moderate absolute errors while making a few larger squared errors. In the corrected fixed-split experiment, hidden RF improves MAE relative to the mean baseline but worsens RMSE and R². The expanded benchmark is a separate comparison, where no learned model beats the median on OOF MAE.

## Q18. What is data leakage in this project?

**Answer.** It is inappropriate access to evaluation information during feature construction, fitting or selection. Examples include using generated answers as pre-generation features, fitting PCA before splitting all prompts, putting the same prompt in training and testing, or tuning a threshold on the targets later used to evaluate it. Leakage can make performance appear better than it is.

## Q19. How does nested CV differ from a fixed split?

**Answer.** A fixed split holds one set out for testing. Nested CV repeats outer testing and uses only the corresponding outer training data for inner model and policy selection. The outer result evaluates the selection procedure. It must not be reused to choose a global winner and then reported as an unbiased score for that winner.

## Q20. Does the saved benchmark prove nested selection was correct?

**Answer.** Not completely. The outer memberships, OOF mappings, metrics and per-fold parameter dictionaries are saved. The original runner, inner-validation predictions and complete fit lineage were not found. That is enough for numerical checks and partial recovery, but not enough to certify every preprocessing and selection boundary. It is not proof that leakage occurred everywhere either.

## Q21. Why is missing selection lineage important for routing?

**Answer.** To choose a threshold for outer fold A, validation scores must be produced without A's outcomes influencing their fit or selection. Ordinary OOF scores from fold B may have been fitted using A's outcomes. Removing A's rows from the tuning table does not remove this indirect dependence. Outer-specific validation lineage is therefore needed.

## Q22. What did the recovery audit actually find?

**Answer.** It found 165 saved per-fold hyperparameter-choice entries and verified 195 fold MAEs from OOF rows. The checkpoint duplicates final results, while the best-result file is a global OOF ranking. Older serialized Ridge models belong to smaller batch splits, not the required outer models. Missing inner predictions and threshold records were not reconstructed or fabricated.

## Q23. What does the 176-policy routing evaluation establish?

**Answer.** It shows retrospective sensitivity of escalation, retained empirical risk and unreliable-mass capture to a fixed threshold grid. It includes all eligible configurations, paired intervals and non-predictive references. It does not select a validated model and threshold, establish real verification effectiveness or demonstrate a reduction in unreliable answers in a deployed system.

## Q24. Why exclude out-of-bounds predictions rather than clip them?

**Answer.** The risk interface requires finite values in the valid range. Clipping after observing results would change the estimator and could hide model failure. The analysis excludes each violating configuration as a whole, retains the same prompts for all reported policies, and records the exclusions. A clipped estimator could be studied only as an explicitly specified separate method.

## Q25. Is a value between zero and one calibrated?

**Answer.** No. Boundedness is a mathematical output constraint. Calibration asks whether predictions at a given level correspond to observed frequencies on appropriate held-out data. A constant or clipped number can be bounded and still misrepresent risk. The current output explicitly says calibration is not established and provides no individual uncertainty interval.

## Q26. What do the bootstrap intervals mean?

**Answer.** They describe aggregate differences when prompts are resampled under the documented scheme. Pairing compares methods on the same sampled identities. Routing intervals hold fitted scores and masks fixed and resample within outer folds. They do not include all training variation, repair leakage, prove individual correctness or automatically adjust for many comparisons.

## Q27. Why is zero risk not proof that the next answer is safe?

**Answer.** It means no unreliable judgment was observed among ten attempts, or, for the median baseline, simply that the training median was zero. Neither interpretation proves a latent failure probability of zero. The next answer is stochastic, and the judgments are imperfect. The conservative default policy still escalates a zero median prediction.

## Q28. Is LUX production-ready?

**Answer.** The baseline Python contract has meaningful tests, explicit artifacts and conservative failure handling. That is useful engineering readiness within a narrow scope. It is not production validation of an individualized risk detector, a network service or a beneficial routing intervention. Learned deployment, independent validation and sustained operational measurements remain incomplete.

## Q29. How could LUX be integrated into a RAG pipeline?

**Answer.** A future application could estimate a prompt's risk and select normal answering, additional retrieval or verification. If retrieval changes the context, the input has changed and the estimate's applicability must be reconsidered. The current constant interface would route all prompts alike under fixed thresholds, so useful selective RAG integration requires a validated learned decision rule and downstream testing.

## Q30. How would you evaluate downstream reliability?

**Answer.** Compare ordinary generation, LUX-informed routing and a simple non-predictive policy at matched intervention budgets. Record actual unreliable delivered answers, false reassurance, coverage, abstention, intervention rate, latency and compute cost. Keep prompt groups and failures visible. Hypothetical perfect verification is not a substitute for observing what the intervention actually produces.

## Q31. What is the strongest defensible contribution?

**Answer.** The strongest contribution is an inspectable empirical investigation that combines frozen-target outcome construction, pre-generation feature comparisons, strong constant baselines and explicit methodological auditing. The negative result is useful because it shows why a plausible representation and a low metric alone are insufficient evidence for practical risk-sensitive decisions.

## Q32. What would you improve first with more time?

**Answer.** I would execute a small, fully recorded nested model and threshold-selection experiment after defining the operational costs. I would preserve outer folds, generate inner validation scores safely and save the complete lineage. Only after that would I justify richer features, calibration, broader target models and a downstream intervention study.

## Q33. Why does cold start take seconds when warm inference takes a fraction of a millisecond?

**Answer.** The supported predictor is a constant. Once imported, it mainly validates an artifact and returns a value. The package initializer eagerly imports the older predictor stack, which dominates the saved cold measurement. The probe did not load Qwen. A hidden-state predictor would need its own model-loading, forward-pass and memory measurements.

## Q34. Are the reported test counts current tests rerun for this report?

**Answer.** No. They are recorded outcomes from the engineering milestones and are labeled by snapshot. This task is documentation only. Numerical research and timing claims were cross-checked by reading existing artifacts, while the historical test counts were checked against the recorded reports and corresponding test code. They were not turned into a new execution claim.

## Q35. How would you choose a threshold without inventing one?

**Answer.** First specify the action and its costs, including the harm of an unreliable delivered answer and the burden of escalation. Then select the model and threshold only inside each outer training partition using inner validation predictions. Freeze that decision before outer testing. If costs or intervention effectiveness are unknown, report sensitivity and do not declare a validated operational threshold.

## Q36. Does the mean baseline contradict the decision to retain the median comparator?

**Answer.** No. They estimate different constants for different loss functions. The mean is a natural cohort expected-risk reference; the median minimizes absolute error. Both are reported, and neither ranks prompts. Exposing the mean in the example interface does not erase its unfavorable MAE or convert it into an individualized calibrated predictor.

# Evidence register and final verification

## Primary source map

The report is readable independently, but the following repository paths identify the evidence used. Paths and code identifiers retain their technical punctuation.

| Subject | Primary evidence |
|---|---|
| Canonical normalization and cohort selection | `src/dataset/squad_adapter.py`, `src/production/expanded_cohort.py`, `data/production/squad_v2_expanded_500/selection.json` |
| Frozen rendering, seeds and hidden extraction | `run_cp4_2_pilot.py`, `run_production.py`, `data/production/squad_v2_expanded_500/generations.json` |
| Evaluator meanings and version | `src/evaluation/prompt.py`, `src/evaluation/prompt_cp4_7.py`, `data/production/cp4_7_0_gemini_3_5_flash_lite/targets_batch_05.json` |
| Cohort matrices and original split | `experiments/final_cohort_2026_10_24/` |
| Original fixed-split metrics | `experiments/final_cohort_2026_10_24/model_comparison.json` |
| Corrected CV source and results | `cv_run_proper.py`, `experiments/cv_model_comparison_2026_10_24/experiment_results.json` and saved test arrays |
| Expanded OOF results and partial parameter lineage | `experiments/expanded_model_selection_2026_10_24/nested_cv_results.json`, `fold_assignments.json`, `checkpoint.json`, `resume_opt.log` |
| Recomputed regression metrics | `experiments/expanded_model_selection_2026_10_24/verification/verification_report.md`, `docs/engineering/oof_analysis_v2.json` |
| Routing metrics, assumptions and intervals | `experiments/routing_policy_evaluation_2026_10_27/results.json`, `analysis.py`, `report.md` |
| Supported interface and export provenance | `src/predictor/inference.py`, `docs/engineering/export_baselines.py`, `baseline_mean_v2.json`, `baseline_median_v2.json` |
| Engineering behavior and recorded test outcomes | `tests/test_risk_inference.py`, `tests/test_inference_end_to_end.py`, `tests/test_saved_oof_analysis.py`, `tests/test_baseline_export.py`, engineering implementation and validation reports |
| Operational raw measurements | `docs/engineering/milestone1_measurements_v2.json`, `measure_baseline_inference.py` |
| Audit and recovery conclusions | `docs/engineering/audit_2026_10_27.md`, `implementation_report.md`, `end_to_end_validation.md`, `policy_selection_recovery_audit.md` |

The files in a shared directory are identified in the table relative to the first fully specified path where necessary. No external literature claim or state-of-the-art comparison is asserted from this register.

## Numerical evidence limits

The fixed-split table's recorded metrics are available, but its complete per-model row predictions are not all persisted there. The corrected experiment's selected test predictions and the expanded benchmark's OOF predictions are persisted and can be reconciled. The full expanded inner-search scores, grids, fit lineage and original execution environment cannot be verified from the recovered files. No calibrated error rate, individual confidence coverage, validated policy threshold or actual downstream reliability improvement is available to report.

Historical test counts have report-level evidence and corresponding test implementations. They are not independently reproduced command logs from this documentation task. The operational measurements have raw JSON observations, but no production generalization claim is made. Early feasibility reports contain other target-model resource figures; they are not mixed into the baseline timing table because their measurement scopes differ.

## Review-preparation checks

The final preparation checks reconcile the stated generation and evaluation counts, target distribution, saved regression metrics, selected routing examples and raw timing summaries. Model-selection evidence is described as partial, not absent and not complete. The architecture distinguishes research-only feature extraction from supported constant inference and marks every downstream integration as future work.

The report contains all 14 requested chapters and 36 answered examiner questions. The diagram uses a plain-text code block with no renderer dependency. Em dashes are excluded, and slash characters are retained only in technical expressions such as paths, identifiers and code formulas. Historical data and experiment files are checked for unchanged file sizes and modification times, with direct source hashes checked for the principal analyzed inputs. No source code, configuration, historical result or model artifact is changed by this report.
