# LUX — Language Uncertainty Extractor

## Engineering interface (2026-10-27)

The historical status below describes an earlier phase. The saved production cohort
contains 500 prompts / 5,000 generations; 486 pass strict eligibility. Learned models
have not demonstrated consistent MAE improvement over the median baseline. The new
interface exposes cohort baselines without claims of prompt-specific discrimination,
calibration, or hallucination prevention.

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
. .\.venv\Scripts\Activate.ps1
python -c "import sys; print(sys.executable)"
python -m pytest tests/test_risk_inference.py -q
```

Use the engineering artifact without downloading or loading a target model:

```python
from src.predictor.inference import (
    Artifact, GENERATION_CONFIG, MODEL, REVISION, Policy, predict_risk,
)

artifact = Artifact.load("docs/engineering/baseline_mean_v2.json")
result = predict_risk(
    {"question": "What is the capital?", "context": "The supplied passage..."},
    MODEL, REVISION, dict(GENERATION_CONFIG), artifact=artifact,
)
print(result.to_dict())
print(Policy().route(result))  # conservative default: escalate
```

The mean artifact returns approximately 0.2086 for every accepted prompt. The
median artifact returns zero for every prompt: useful for MAE comparison, never proof
of safety. Outputs include predictor fingerprint, target revision, pipeline version,
warnings, `calibrated=False` and `uncertainty=None`. Logs omit prompt content.
Inputs accept text or exactly question/context; enriched generation records and
unsupported settings are rejected. Out-of-domain text cannot be reliably detected;
all outputs warn about external validity. The character limit is a resource guard.

`Policy(verify_at=..., escalate_at=...)` accepts caller-selected thresholds; none
are validated or supplied as defaults. Uncalibrated routing also requires explicit
`allow_uncalibrated=True`. Missing estimates or thresholds escalate. A constant
baseline routes all prompts alike. Catch artifact/input failures and use
`Policy().route(None)` for a conservative action. The layer does not execute actions.

Artifacts use strict versioned JSON with provenance hashes and exclusive creation.
Reproduce export with `python -m docs.engineering.export_baselines` in a fresh copy
without the output files; existing artifacts cannot be overwritten. Constants are
fitted on all eligible data for deployment, not evaluated as a new held-out benchmark.
No target-model resource or hidden-extraction latency claim is made.

See [audit](docs/engineering/audit_2026_10_27.md),
[plan and future evaluation protocol](docs/engineering/implementation_plan.md),
and [implementation report](docs/engineering/implementation_report.md).
No external validation or downstream reliability benefit has been demonstrated.

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
