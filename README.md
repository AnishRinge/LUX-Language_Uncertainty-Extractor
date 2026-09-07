# LUX — Language Uncertainty Extractor

**Pre-Generation LLM Hallucination Risk Prediction**

LUX is a research project investigating whether an LLM's **pre-generation internal representations**, combined with observable **prompt-level features**, can predict the likelihood that the model will produce an unreliable response.

## Research Question

> Can combining observable prompt-level features with the target LLM's internal pre-generation hidden representations improve hallucination-risk prediction compared with using either information source alone?

## Core Constraints

- **Pre-generation prediction** — risk is estimated before the target LLM generates an answer.
- **Frozen target LLM** — the target model is not fine-tuned.
- **Separate predictor** — the risk predictor is separate from the target LLM.
- **No answer leakage** — generated answers are used only to construct empirical ground-truth targets, never as predictor features.
- **Baseline comparison** — prompt-only, hidden-state-only, and fused predictors will be compared.

---

## Project Status

### Phase 0 — Dataset Foundation: COMPLETE

Phase 0 established and validated the dataset infrastructure and canonical prompt-scenario representation.

| Checkpoint | Status |
|---|---|
| CP1 — Project Specification | PASS |
| CP2 — Dataset Creation Methodology | APPROVED |
| CP3.1 — Dataset Infrastructure | PASS |
| CP3.2 — SQuAD Artifact Validation | PASS |
| CP3.3 — SQuAD Suitability + Canonical Schema | APPROVED |
| CP3.4 — SQuAD Adapter + Normalization | PASS |
| CP3.5 — Integrity, Duplicate & Leakage Audit | PASS |
| CP3.6 — Pilot Readiness Gate | READY_FOR_PILOT |

### Current Dataset

The current repository contains a **canonical prompt-scenario dataset derived from SQuAD v2**.

- 130,319 normalized scenarios
- 86,821 answerable
- 43,498 unanswerable
- Raw source artifact validated
- Canonical schema implemented
- Integrity and leakage checks completed

The canonical dataset is kept locally and is **not tracked in Git**.

**The final LUX hallucination-risk dataset does not yet exist.**

Qwen generations, generation-level evaluations, empirical risk targets, hidden-state representations, and predictor features will be created in later phases.

---

## Architecture

```text
SQuAD / Future Datasets
        │
        ▼
Canonical Prompt Scenarios
        │
        ├──────────────► Prompt-Level Features
        │
        ▼
   Frozen Qwen3
        │
        ▼
Pre-Generation Hidden Representation
        │
        └──────────────┐
                       ▼
              Hallucination-Risk
                  Predictor
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
   Prompt-only    Hidden-only       Fusion
     baseline       baseline        model