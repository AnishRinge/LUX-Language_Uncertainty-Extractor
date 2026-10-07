"""CP4.4 evaluator prompt construction and response schema.

This module owns the frozen evaluator instructions and the Gemini
structured-output JSON schema. It depends only on the standard library so it
can be imported and unit-tested without the Google GenAI SDK.

The evaluator receives ONLY:
  1. the SQuAD question/prompt
  2. the supplied SQuAD context
  3. the generated Qwen3 answer

It must not receive Qwen3 hidden states, logits, sampling metadata, random
seed, target-model confidence, or any information about whether the answer
came from Qwen3.
"""

import json
from typing import Any, Dict, List

from .schemas import (
    ClaimStatus,
    FailureType,
    GenerationLabel,
    EvaluationStatus,
    Materiality,
)

PROMPT_VERSION = "CP4.4.0"
EVALUATOR_MODEL = "gemini-3.8-flash"
DEFAULT_TEMPERATURE = 0.2

EVALUATOR_SYSTEM_INSTRUCTION = """You are an independent evidence-grounded evaluator.

Your task is to evaluate a generated response using ONLY:
- the question/prompt
- the supplied context
- the generated response

Do not assume information that is not supported by the supplied evidence.

For each substantive claim:
- SUPPORTED = evidence directly supports the claim
- CONTRADICTED = evidence directly conflicts with the claim
- UNSUPPORTED = the claim goes beyond what the applicable evidence supports and there is sufficient evidence to establish this
- UNVERIFIABLE = the claim cannot be reliably determined from the available evidence
- NOT_APPLICABLE = claim-level evaluation does not meaningfully apply

Do not treat UNVERIFIABLE as UNSUPPORTED.

Do not automatically classify a generation as UNRELIABLE because of one minor or
peripheral issue. A problematic individual claim does not automatically make the
whole generation UNRELIABLE. Generation-level materiality is qualitative; do not
attach arbitrary numeric thresholds.

Classify the whole generation as:
- RELIABLE when no material reliability failure is established and the response
  adequately addresses the task
- UNRELIABLE when at least one material reliability failure is established
- INADEQUATE when the response is incomplete/problematic but a material
  reliability failure cannot be established confidently

When uncertain about materiality or the generation-level decision, use INADEQUATE
and/or request human review according to the schema.

A claim that is UNVERIFIABLE is never treated as a material reliability failure by
itself. A NON_MATERIAL claim (formatting, phrasing, peripheral wording) is never
sufficient on its own to make a generation UNRELIABLE.

For SQuAD v2: the supplied passage is the primary evidence source. The passage is
the ONLY evidence you may consult. is_impossible=true means the answer is
unavailable from the supplied context; it does NOT let you infer global
unknowability, and it does NOT authorise you to use external knowledge to answer.

You are NOT the target model and you must not reproduce or re-rank the answer.
"""


def build_user_message(question: str, context: str, generated_answer: str) -> str:
    """Assemble the user message carrying only question, context and answer."""
    return (
        "Question:\n{question}\n\n"
        "Supplied context:\n{context}\n\n"
        "Generated response:\n{answer}\n\n"
        "Evaluate using only the question, the supplied context, and the generated "
        "response. Return your evaluation as JSON matching the required schema."
    ).format(question=question, context=context, answer=generated_answer)


def build_messages(question: str, context: str, generated_answer: str) -> List[Dict[str, str]]:
    """System + user message list for the Gemini call."""
    return [
        {"role": "system", "content": EVALUATOR_SYSTEM_INSTRUCTION},
        {"role": "user", "content": build_user_message(question, context, generated_answer)},
    ]


def response_schema() -> Dict[str, Any]:
    """Gemini-compatible JSON schema for the structured evaluator output.

    The ``evaluation_status`` field lets the model signal whether it resolved the
    evaluation (SUCCESS) or requires a human because of semantic uncertainty
    (MANUAL_REVIEW). Transport outcomes (RETRY / FAILED) are owned by the
    wrapper and are never produced by the model.
    """
    return {
        "type": "object",
        "properties": {
            "evaluation_status": {
                "type": "string",
                "enum": [s.value for s in EvaluationStatus if s.value in ("SUCCESS", "MANUAL_REVIEW")],
            },
            "claims": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "claim_text": {"type": "string"},
                        "claim_status": {
                            "type": "string",
                            "enum": [s.value for s in ClaimStatus],
                        },
                        "evidence_text": {"type": "string"},
                        "failure_type": {
                            "type": "string",
                            "enum": [f.value for f in FailureType],
                        },
                        "materiality": {
                            "type": "string",
                            "enum": [m.value for m in Materiality],
                        },
                    },
                    "required": [
                        "claim_text",
                        "claim_status",
                        "evidence_text",
                        "failure_type",
                        "materiality",
                    ],
                },
            },
            "generation_label": {
                "type": "string",
                "enum": [g.value for g in GenerationLabel],
            },
            "generation_failure_types": {
                "type": "array",
                "items": {"type": "string", "enum": [f.value for f in FailureType]},
            },
            "rationale": {"type": "string"},
            "needs_human_review": {"type": "boolean"},
        },
        "required": [
            "evaluation_status",
            "claims",
            "generation_label",
            "generation_failure_types",
            "rationale",
            "needs_human_review",
        ],
    }


def stringify_schema() -> str:
    """Return the response schema as a JSON string (for logging/debug only)."""
    return json.dumps(response_schema(), indent=2)
