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

## 17. Checkpoint 2 Gate
**UNRESOLVED**: Checkpoint 2 is Dataset Selection. 

The gate requires us to finalize which QA dataset(s) will actually be used and the rationale for that selection BEFORE dataset ingestion begins. No dataset ingestion or feature extraction should begin before that decision is approved. (NOT STARTED)


