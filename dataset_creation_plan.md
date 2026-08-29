# Dataset Creation and Ground-Truth Construction Plan (Checkpoint 2 Conceptual Specification)

**Status**: CP2 conceptual specification  
**Purpose**: Authoritative planning document for dataset creation and ground-truth construction  
**Implementation status**: NOT STARTED  
**Dataset ingestion status**: NOT STARTED  
**Qwen generation status**: NOT STARTED  

*This document records approved research direction and unresolved decisions. It does not authorize implementation of unresolved decisions.*

---

## 1. Project Objective

LUX investigates:

> "Can combining observable prompt-level features with the target LLM's internal pre-generation hidden representations improve hallucination-risk prediction compared with using either information source alone?"

The core research objective is NOT merely to predict whether an LLM answers factual QA incorrectly. 

LUX is intended to investigate the broader propensity of modern LLMs to produce unreliable/hallucinatory responses, including situations where the model:
- fills knowledge gaps instead of expressing uncertainty,
- accepts misleading premises,
- confidently resolves ambiguity without justification,
- makes unsupported inferences,
- fabricates plausible entities,
- fabricates research papers/citations/references,
- contradicts or goes beyond supplied context,
- fails in multi-hop reasoning,
- produces plausible but unsupported open-ended factual claims.

**Crucial Constraints**:
- The prediction must occur **BEFORE** the target model generates its answer.
- Generated answers must **NEVER** be used as predictor features.

---

## 2. Approved Hallucination / Unreliability Taxonomy

The current approved LUX phenomena comprise:

1. **Context contradiction**
2. **Context unsupportedness**
3. **Knowledge uncertainty / knowledge gaps**
4. **Nonexistent entity fabrication**
5. **Misleading-premise susceptibility**
6. **Ambiguity / underspecification**
7. **Unsupported inference**
8. **Multi-hop reasoning failure**
9. **Fabricated citations/references**
10. **Open-ended factual unreliability**

*Note*: This taxonomy is broader than conventional factual QA hallucination detection. Not all of these phenomena are equally represented by existing benchmark datasets.

---

## 3. Dataset Strategy

The current approved dataset roles are defined as follows:

### CORE CANDIDATES:
- **SQuAD 2.0**
  - *Purpose*: Grounded/contextual reliability, including answerable and unanswerable questions.
- **HalluLens**
  - *Priority areas*: PreciseWikiQA, NonExistentRefusal, LongWiki where appropriate.
  - *Purpose*: Knowledge gaps, factual reliability, nonexistent entities/fabrication.
- **TruthfulQA**
  - *Purpose*: Misleading premises, misconceptions, truthfulness.

### GENERALIZATION / CHALLENGE CANDIDATES:
- **HotpotQA**
  - *Purpose*: Multi-hop reasoning.
- **Natural Questions**
  - *Purpose*: Realistic information-seeking queries.
- **FactBench**
  - *Purpose*: Open-ended/in-the-wild factual reliability.
- **HaluEval**
  - *Purpose*: Supplementary/generalization only, with strict leakage precautions.

### SPECIAL CUSTOM COMPONENT:
- A possible **LUX-specific fabricated citation/reference component**.
- Intended to cover cases such as *"Find research papers about XYZ"* where a model may invent plausible-sounding papers, authors, venues, DOIs, etc.
- *Note*: This custom component has NOT been created yet. Exact construction method, size, and sources remain **UNRESOLVED**.

### LOW PRIORITY / NOT CORE:
- TriviaQA
- HaluEval dialogue/summarization
- HaluEval general-response labels as direct LUX ground truth

*Important*: These roles represent approved research direction, but exact sample counts and final dataset membership remain unresolved until the relevant later checkpoint.

---

## 4. Dataset Structure

The proposed three-layer conceptual structure:

1. **CORE**
   - Used for primary LUX development/training.
2. **GENERALIZATION**
   - Used to test whether the learned signal transfers across different datasets/phenomena.
3. **CHALLENGE**
   - Used for difficult real-world / open-ended scenarios, including the possible custom fabricated-citation component.

*Note*: Exact dataset membership within these categories is not permanently frozen.

---

## 5. LUX Sample Concept

The fundamental unit is a **PROMPT SCENARIO**, not merely a question.

Conceptual sample fields:
- `sample_id`
- `source_dataset`
- `source_split`
- `phenomenon`
- `prompt`
- `context` (optional)
- `reference information`
- `answerability` (where applicable)
- `expected behavior` (where applicable)
- `difficulty` (where available/defined)
- `metadata`

*Crucial Constraint*: `source_dataset` and `phenomenon` are metadata for analysis and dataset management. They **MUST NOT** be provided as predictor features.

---

## 6. Prompt / Generation Separation

Static prompt data and generated responses must be stored separately.

Conceptual structure:

- **PROMPT RECORD**:
  - `sample_id`, `prompt`, `context`, `references`, `metadata`
- **GENERATION RECORD**:
  - `sample_id`, `generation_id`, `model`, `model revision`, `generation configuration`, `response`
- **EVALUATION RECORD**:
  - `sample_id`, `generation_id`, `generation-level outcome`, `diagnostic failure type`, `claim-level evaluations` (where applicable), `evaluator information`, `evidence information` (where applicable)
- **TARGET RECORD**:
  - `sample_id`, `number of generations`, `number reliable`, `number unreliable`, `number inadequate`, `empirical unreliability risk`

Generated responses and evaluation outputs are for **TARGET CONSTRUCTION ONLY**. They must never become predictor input features.

---

## 7. Generation-Level vs. Prompt-Level

- A **generation-level record** represents ONE individual response produced by Qwen for one prompt.
- A **prompt-level target** is obtained by repeatedly generating responses for the same prompt and aggregating the generation-level reliability outcomes.

*Example*:
Prompt X:
- Generation 1 → RELIABLE
- Generation 2 → UNRELIABLE
- Generation 3 → RELIABLE
- Generation 4 → UNRELIABLE
- Generation 5 → UNRELIABLE

Empirical risk: 3 / 5 = 0.60

The exact number of generations $N$ is still **UNRESOLVED** (do not assume $N=5$ or $N=10$).

---

## 8. Generation-Level Label Ontology

Approved conceptual generation outcomes:
- **RELIABLE**: The response is substantively correct and/or appropriately handles the prompt.
- **UNRELIABLE**: The response contains a material hallucination/unreliability that materially affects the answer or violates required evidence/context.
- **INADEQUATE**: The response is poor or fails to fulfill the task but is not necessarily a hallucination (e.g., inappropriate refusal on an answerable question).

*Note*: `INADEQUATE` is recorded for analysis but does **NOT** count as an unreliable generation in the current proposed risk formulation. Do not silently convert `INADEQUATE` into `UNRELIABLE`.

---

## 9. Claim-Level Label Ontology

Approved claim-level statuses:
- **SUPPORTED**
- **CONTRADICTED**
- **UNSUPPORTED**
- **UNVERIFIABLE**
- **NOT_APPLICABLE**

*Important*: `UNVERIFIABLE` does **NOT** automatically mean hallucination. "Not verified" is not equivalent to "false." Surrounding benchmark evidence and adjudication rules determine whether a generation becomes unreliable.

---

## 10. Behavior Labels

Conceptual behavior labels:
- **ANSWER**
- **ABSTAIN**
- **CLARIFY**
- **QUALIFY**
- **REFUTE_PREMISE**

*Important*: Refusal/abstention is not automatically good. Refusing an answerable factual question may be `INADEQUATE`. Appropriate abstention/qualification/clarification/refutation may be `RELIABLE` when the prompt warrants it.

---

## 11. Diagnostic Failure Types

Approved diagnostic failure types:
- **NONE**
- **FACTUAL_ERROR**
- **CONTEXT_CONTRADICTION**
- **CONTEXT_UNSUPPORTED**
- **FABRICATED_ENTITY**
- **FABRICATED_CITATION**
- **MISLEADING_PREMISE_ACCEPTANCE**
- **UNSUPPORTED_INFERENCE**
- **REASONING_ERROR**
- **AMBIGUOUS_INTERPRETATION**
- **OTHER**

These labels are diagnostic and must be retained separately from the final risk target.

---

## 12. Materiality Principle

A generation is **UNRELIABLE** when it contains at least one substantive claim or behavioral failure that makes the response materially misleading, false, unsupported, or inconsistent with information the model was expected to respect.

- Do NOT treat every tiny imprecision as a hallucination.
- A material error substantially affects factual meaning, conclusion, recommendation, requested information, or the user's likely understanding.
- The exact operational materiality threshold remains **UNRESOLVED**.

---

## 13. Multi-Claim Responses

Responses may contain multiple substantive claims:
$$\text{Generation} \longrightarrow \text{Claim extraction} \longrightarrow \text{Claim-level verification} \longrightarrow \text{Materiality assessment} \longrightarrow \text{Generation-level outcome}$$

- Do NOT use simple "number of correct claims / total claims" as the final generation label.
- A single material fabrication can make the entire generation `UNRELIABLE`.

---

## 14. Citation / Reference Fabrication

Fabricated citations/references are a first-class LUX phenomenon.
For generated citations, future evaluation may examine title, authors, venue, year, DOI, URL, existence, and bibliographic consistency.

Potential conceptual statuses:
- **EXISTS**
- **DOES_NOT_EXIST**
- **EXISTS_BUT_DETAILS_WRONG**
- **UNVERIFIABLE**

*Constraints*: Do not implement this now. Do not assume Crossref is the sole source. Exact citation verification implementation is **UNRESOLVED**.

---

## 15. Evaluation Architecture

Approved high-level architecture:
$$\text{Qwen generation} \longrightarrow \text{response parser} \longrightarrow \text{claim extraction} \longrightarrow \text{routing} \longrightarrow \text{verification} \longrightarrow \text{claim results} \longrightarrow \text{materiality} \longrightarrow \text{generation outcome} \longrightarrow \text{empirical risk}$$

- The evaluator should be modular rather than a single universal "hallucination judge."
- Preferred evaluation hierarchy:
  1. Deterministic/reference-based checks.
  2. Evidence-based verification.
  3. LLM-assisted semantic evaluation.
  4. Human validation for a subset.
- An LLM must **NOT** automatically be treated as unquestioned ground truth.
- The target Qwen3 model must **NOT** be used as its own evaluator.
- Exact evaluator models remain **UNRESOLVED**.

---

## 16. Human Validation

- A human-adjudicated subset will be used to validate the automated evaluation pipeline across datasets, phenomena, evaluator confidence, ambiguous cases, UNVERIFIABLE cases, partial correctness, and citation fabrications.
- Exact validation sample size is **UNRESOLVED**.
- Human validation is for evaluator calibration, not necessarily for labeling the entire dataset.

---

## 17. Empirical Risk

Approved formulation:
$$\hat{R}(x) = \frac{N_{\text{unreliable}}(x)}{N_{\text{all\_evaluated\_generations}}(x)}$$

Raw counts to store:
- `n_generations`
- `n_reliable`
- `n_unreliable`
- `n_inadequate`
- `empirical_unreliability_risk`

*Constraint*: Do NOT change the denominator to exclude inadequate generations unless explicitly revisited. The exact generation count $N$ remains **UNRESOLVED**.

---

## 18. Leakage Constraints

The following must **NEVER** be predictor input features:
- Generated answers
- Generation-level labels
- Evaluator outputs
- Evidence retrieved after generation
- Empirical risk target
- `source_dataset`
- `phenomenon`
- Any post-generation information

- The predictor is strictly **pre-generation**.
- The target model Qwen3 is **frozen** and **NOT fine-tuned**.
- Layer selection, feature selection, and hyperparameter selection must not use the final test set.

---

## 19. Core vs. Generalization Experiment

- Train/develop primarily using approved core datasets.
- Evaluate generalization on held-out datasets (e.g., HotpotQA, Natural Questions, FactBench) to test whether LUX learns genuine unreliability propensity rather than dataset-specific shortcuts.
- Exact split is **UNRESOLVED**.

---

## 20. Sampling

- We will NOT automatically ingest every available example from every dataset.
- Preliminary planning estimates considered ~10K–15K unique core prompts, but this is **NOT FINAL**.
- A smaller pilot must be used before large-scale target generation.
- Exact dataset sizes, sampling ratios, stratification rules, pilot size, and generation count remain **UNRESOLVED**.

---

## 21. Implementation Strategy

Staged evaluator development:
A. Evaluation infrastructure  
B. SQuAD 2.0 evaluator  
C. TruthfulQA evaluator  
D. HalluLens evaluator  
E. Evidence-based evaluator  
F. Citation/reference verification  
G. Human validation  
H. Full target generation  

The evaluator must be validated before full-scale target generation. Do not generate a large target corpus before evaluator validation.

---

## 22. Reproducibility

Every generation/evaluation target should eventually record:
- Target model & revision / checkpoint
- Tokenizer revision
- Generation configuration & random seed policy
- Evaluator version
- Evidence/retrieval information
- Dataset version
- Processing version

External evidence should be cached. Evaluation failures must not silently become `UNRELIABLE` labels (statuses: `SUCCESS`, `RETRY`, `MANUAL_REVIEW`, `FAILED`). Exact retry/fallback policy is **UNRESOLVED**.

---

## 23. Explicitly Unresolved Decisions

The following items are explicitly **UNRESOLVED** and must not be filled in by assumption:
- Exact Qwen3 checkpoint
- Exact Qwen3/tokenizer revisions
- Exact generation count $N$
- Exact generation parameters
- Exact claim extraction method/model
- Exact semantic evaluator
- Exact retrieval/evidence system
- Exact citation verification sources
- Exact materiality threshold
- Exact ambiguity adjudication rules
- Exact human validation size
- Exact dataset sample counts
- Exact sampling ratios
- Exact train/validation/test split
- Exact evaluator disagreement procedure
- Exact statistical uncertainty method for empirical risk
- Exact custom fabricated-citation dataset construction
