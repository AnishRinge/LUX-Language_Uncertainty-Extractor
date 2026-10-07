"""CP4.4 evaluation infrastructure (independent of the Qwen3 target model).

Public API:
  - schemas: enums, ClaimEvaluation, StructuredEvaluation, EvaluationRecord
  - prompt: frozen system instruction + Gemini structured-output JSON schema
  - retry: configurable RetryPolicy / retry_call / RetryExhausted
  - gemini_evaluator: GeminiEvaluator (lazy SDK import, injectable client)
  - evaluator_runner: EvaluationRunner, build_request_from_cp4_2_5,
    compute_empirical_risk, aggregate_pilot_report
"""

from .schemas import (
    SCHEMA_VERSION,
    EvaluationRecord,
    EvaluationStatus,
    GenerationLabel,
    ClaimStatus,
    ClaimEvaluation,
    StructuredEvaluation,
    FailureType,
    Materiality,
    is_failure_claim,
    infer_generation_label_from_claims,
)
from .prompt import (
    PROMPT_VERSION,
    EVALUATOR_MODEL,
    EVALUATOR_SYSTEM_INSTRUCTION,
    response_schema,
    build_messages,
    build_user_message,
)
from .retry import RetryPolicy, RetryExhausted, RetryableError, retry_call, is_retryable_exception
from .gemini_evaluator import (
    API_KEY_ENV_VAR,
    GeminiEvaluator,
    EvaluationRequest,
    EvaluatorError,
    EvaluatorConfigError,
    TransientEvaluationError,
    PersistentEvaluationError,
    EvaluationParseError,
)
from .evaluator_runner import (
    EvaluationRunner,
    build_request_from_cp4_2_5,
    evaluate_generations,
    compute_empirical_risk,
    aggregate_pilot_report,
)

__all__ = [
    "SCHEMA_VERSION",
    "PROMPT_VERSION",
    "EVALUATOR_MODEL",
    "EVALUATOR_SYSTEM_INSTRUCTION",
    "EvaluationRecord",
    "EvaluationStatus",
    "GenerationLabel",
    "ClaimStatus",
    "ClaimEvaluation",
    "StructuredEvaluation",
    "FailureType",
    "Materiality",
    "is_failure_claim",
    "infer_generation_label_from_claims",
    "response_schema",
    "build_messages",
    "build_user_message",
    "RetryPolicy",
    "RetryExhausted",
    "RetryableError",
    "retry_call",
    "is_retryable_exception",
    "API_KEY_ENV_VAR",
    "GeminiEvaluator",
    "EvaluationRequest",
    "EvaluatorError",
    "EvaluatorConfigError",
    "TransientEvaluationError",
    "PersistentEvaluationError",
    "EvaluationParseError",
    "EvaluationRunner",
    "build_request_from_cp4_2_5",
    "evaluate_generations",
    "compute_empirical_risk",
    "aggregate_pilot_report",
]
