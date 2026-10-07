"""Gemini evaluator for CP4.4.

The evaluator is independent of Qwen3. It receives ONLY the SQuAD
question/prompt, the supplied SQuAD context, and the generated Qwen3 answer. It
never receives Qwen3 hidden states, logits, sampling metadata, the random seed,
or any indication that the answer came from Qwen3.

The Google GenAI SDK is imported lazily (mirroring how ``run_cp4_2_pilot``
lazily imports ``torch`` / ``transformers``) so the module can be imported and
unit-tested without the SDK installed. Tests inject a fake ``client``; a live
run requires ``GEMINI_API_KEY`` and the ``google-genai`` package.

API key handling:
  * Loaded from the ``GEMINI_API_KEY`` environment variable (never hard-coded).
  * Never printed, logged, or embedded in any persisted record.
"""

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Protocol

from .prompt import EVALUATOR_MODEL, PROMPT_VERSION, build_messages, response_schema
from .retry import RetryPolicy, RetryExhausted, RetryableError, retry_call
from .schemas import (
    SCHEMA_VERSION,
    EvaluationRecord,
    EvaluationStatus,
    StructuredEvaluation,
)

API_KEY_ENV_VAR = "GEMINI_API_KEY"
SECRET_PLACEHOLDER = "<GEMINI_API_KEY-redacted>"


class EvaluatorError(Exception):
    """Base evaluator error."""


class EvaluatorConfigError(EvaluatorError):
    """Configuration problem (missing key / SDK). Not retryable."""


class TransientEvaluationError(EvaluatorError, RetryableError):
    """Transient API failure (retryable)."""


class PersistentEvaluationError(EvaluatorError):
    """Persistent, non-retryable API failure (e.g. auth / invalid argument)."""


class EvaluationParseError(EvaluatorError, RetryableError):
    """The model output could not be parsed into a valid schema (retryable)."""


class _FakeResponse:
    """Minimal stand-in for a Gemini response carrying JSON ``text``."""

    def __init__(self, text: str):
        self.text = text


class GeminiClient(Protocol):
    """Structural interface the evaluator depends on for a Gemini-compatible client."""

    models: Any


@dataclass
class EvaluationRequest:
    question: str
    context: str
    generated_answer: str
    source_generation_id: str
    dataset_id: str
    sample_id: str = ""
    generation_index: int = 0
    seed: Optional[int] = None
    model_name: str = ""
    model_revision: str = ""
    answerability: str = ""


class GeminiEvaluator:
    """Evidence-grounded hallucination-risk evaluator backed by Gemini.

    Parameters
    ----------
    model:
        Frozen evaluator model id (``gemini-3.8-flash``).
    api_key:
        Explicit key. If omitted, read from ``GEMINI_API_KEY``. Ignored when
        ``client`` is injected (tests).
    client:
        Injectable Gemini-compatible client. When provided the SDK is never
        imported, so tests need no SDK and no API key.
    retry_policy:
        Configurable ``RetryPolicy`` (retry count is data-driven, not hard-coded).
    """

    def __init__(
        self,
        model: str = EVALUATOR_MODEL,
        api_key: Optional[str] = None,
        client: Optional[Any] = None,
        retry_policy: Optional[RetryPolicy] = None,
        prompt_version: str = PROMPT_VERSION,
        schema_version: str = SCHEMA_VERSION,
        temperature: float = 0.2,
    ):
        self.model = model
        self.prompt_version = prompt_version
        self.schema_version = schema_version
        self.temperature = temperature
        self.retry_policy = retry_policy or RetryPolicy()
        self._injected_client = client
        self._resolved_api_key = (
            api_key if api_key is not None else os.environ.get(API_KEY_ENV_VAR)
        )
        # The API key is deliberately never persisted or logged.

    # ------------------------------------------------------------------
    # Client construction (lazy SDK import)
    # ------------------------------------------------------------------
    def _build_client(self) -> Any:
        """Create a real google.genai.Client. Lazy so tests need no SDK."""
        try:
            from google import genai  # type: ignore
        except Exception as exc:  # noqa: BLE001 - surface a clear config error
            raise EvaluatorConfigError(
                "google.genai SDK is not installed; install with "
                "`pip install google-genai` or inject a client for testing"
            ) from exc
        if not self._resolved_api_key:
            raise EvaluatorConfigError(
                "GEMINI_API_KEY is not set; cannot construct a real Gemini client"
            )
        return genai.Client(api_key=self._resolved_api_key)

    def _client(self) -> Any:
        if self._injected_client is not None:
            return self._injected_client
        return self._build_client()

    # ------------------------------------------------------------------
    # Request building
    # ------------------------------------------------------------------
    def _generation_config(self) -> Dict[str, Any]:
        return {
            "temperature": self.temperature,
            "response_mime_type": "application/json",
            "response_schema": response_schema(),
        }

    # ------------------------------------------------------------------
    # The single Gemini call
    # ------------------------------------------------------------------
    def _call_gemini(self, messages: List[Dict[str, str]]) -> str:
        """Issue one structured-output call and return the raw response text.

        The evaluator models the prompt internally as OpenAI-style message dicts
        (``{role, content}``) for clarity, but the installed google-genai SDK
        does not accept those as the ``contents`` argument (it raises a local
        pydantic ``ValidationError`` before any request is sent). Convert to the
        SDK's native shape instead: the user turn (built solely from
        question/context/generated_answer) becomes a string ``contents``, and the
        system turn becomes ``system_instruction`` in the generation config.
        """
        client = self._client()
        config = dict(self._generation_config())
        system_text = ""
        user_text = ""
        for msg in messages:
            if msg.get("role") == "system":
                system_text = msg.get("content", "")
            elif msg.get("role") == "user":
                user_text = msg.get("content", "")
        config["system_instruction"] = system_text
        try:
            response = client.models.generate_content(
                model=self.model,
                contents=user_text,
                config=config,
            )
        except EvaluatorConfigError:
            raise
        except Exception as exc:  # noqa: BLE001 - classify API errors
            raise self._classify_api_error(exc) from exc
        text = getattr(response, "text", None)
        if text is None:
            parsed = getattr(response, "parsed", None)
            if parsed is not None:
                return json.dumps(parsed)
            raise EvaluationParseError("Gemini response had no text/parsed content")
        return text

    def _redact(self, text: str) -> str:
        """Redact any known API-key material from a string before persistence."""
        if not text:
            return text
        for key in (self._resolved_api_key, os.environ.get(API_KEY_ENV_VAR, "")):
            if key:
                text = text.replace(key, SECRET_PLACEHOLDER)
        return text

    def _classify_api_error(self, exc: BaseException) -> EvaluatorError:
        # Deterministic, local SDK argument-validation failures must not be
        # retried: a malformed request fails identically on every attempt.
        if type(exc).__name__ == "ValidationError" and "validation error" in str(
            exc
        ).lower():
            return PersistentEvaluationError(
                self._redact(
                    "deterministic SDK request-validation error (not retried): {0}".format(
                        exc
                    )
                )
            )
        code = getattr(exc, "code", None)
        code_str = str(code) if code is not None else ""
        name = type(exc).__name__
        if code_str in ("401", "403"):
            return PersistentEvaluationError(
                self._redact("gemini authentication/authorization failed: {0}".format(exc))
            )
        if code_str == "400" or name in ("InvalidArgument",):
            return PersistentEvaluationError(
                self._redact("gemini invalid argument: {0}".format(exc))
            )
        return TransientEvaluationError(
            self._redact("gemini transient failure: {0}".format(exc))
        )

    # ------------------------------------------------------------------
    # Error reporting helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _error_type_for(error: BaseException) -> str:
        """Surface the root-cause exception type for retryable-wrapped errors.

        Retryable API/parse errors are wrapped in ``TransientEvaluationError``
        (with the original preserved as ``__cause__``). For those we report the
        underlying cause type (e.g. ``TimeoutError``) so operators see the real
        failure. Persistent / config errors keep their own wrapper type.
        """
        root = error
        if isinstance(error, TransientEvaluationError):
            cause = getattr(error, "__cause__", None)
            while cause is not None:
                root = cause
                cause = getattr(cause, "__cause__", None)
        return type(root).__name__

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_json(text: str) -> str:
        """Strip surrounding ```json fences and whitespace, if present."""
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.lstrip("`")
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:]
            cleaned = cleaned.strip()
        if cleaned.endswith("```"):
            cleaned = cleaned.rstrip("`").strip()
        return cleaned

    def _parse(self, text: str) -> StructuredEvaluation:
        try:
            data = json.loads(self._extract_json(text))
        except ValueError as exc:
            raise EvaluationParseError(
                "evaluator output was not valid JSON: {0}".format(exc)
            )
        if not isinstance(data, dict):
            raise EvaluationParseError("evaluator output JSON was not an object")
        try:
            return StructuredEvaluation.from_dict(data)
        except ValueError as exc:
            raise EvaluationParseError(
                "evaluator output did not conform to schema: {0}".format(exc)
            )

    # ------------------------------------------------------------------
    # Status resolution
    # ------------------------------------------------------------------
    def _resolve_status(self, structured: StructuredEvaluation) -> EvaluationStatus:
        if structured.needs_human_review:
            return EvaluationStatus.MANUAL_REVIEW
        if structured.evaluation_status == EvaluationStatus.MANUAL_REVIEW:
            return EvaluationStatus.MANUAL_REVIEW
        return EvaluationStatus.SUCCESS

    # ------------------------------------------------------------------
    # Public evaluation
    # ------------------------------------------------------------------
    def evaluate(self, request: EvaluationRequest) -> EvaluationRecord:
        """Evaluate one generation with retry. Returns a terminal record.

        Terminal ``evaluation_status`` is one of:
          * SUCCESS        - resolved; carries a generation_label.
          * MANUAL_REVIEW  - parsed but flagged uncertain.
          * FAILED         - transport/parse failure exhausted retries. NEVER
                               carries a generation_label and NEVER contributes
                               to empirical risk.
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        schema = response_schema()
        messages = build_messages(
            request.question, request.context, request.generated_answer
        )
        attempts: List[Dict[str, Any]] = []
        policy = self.retry_policy

        def _one_attempt() -> StructuredEvaluation:
            raw = self._call_gemini(messages)
            return self._parse(raw)

        def _log(attempt_number: int, exc: Optional[Exception]) -> None:
            if exc is None:
                attempts.append({"attempt": attempt_number, "status": "SUCCESS"})
            else:
                retryable = policy.is_retryable(exc) if policy else False
                attempts.append(
                    {
                        "attempt": attempt_number,
                        "status": "RETRY" if retryable else "FAILED",
                        "error_type": type(exc).__name__,
                        "error_message": self._redact(str(exc)),
                    }
                )

        try:
            structured = retry_call(
                _one_attempt,
                policy=policy,
                on_attempt=_log,
                description="gemini evaluation of {0}".format(
                    request.source_generation_id
                ),
            )
        except RetryExhausted as exc:
            retry_count = max(0, len(attempts) - 1)
            return self._failed_record(
                request, timestamp, attempts, retry_count, exc.last_error
            )
        except PersistentEvaluationError as exc:
            return self._failed_record(
                request, timestamp, attempts, retry_count=max(0, len(attempts) - 1), error=exc
            )

        retry_count = max(0, len(attempts) - 1)
        terminal_status = self._resolve_status(structured)
        record = EvaluationRecord.from_structured(
            structured,
            evaluator_model=self.model,
            prompt_version=self.prompt_version,
            schema_version=self.schema_version,
            source_generation_id=request.source_generation_id,
            dataset_id=request.dataset_id,
            retry_count=retry_count,
            timestamp=timestamp,
            attempts=attempts,
        )
        record.evaluation_status = terminal_status
        record.needs_human_review = bool(
            structured.needs_human_review
            or terminal_status == EvaluationStatus.MANUAL_REVIEW
        )
        return record.validate()

    def _failed_record(
        self,
        request: EvaluationRequest,
        timestamp: str,
        attempts: List[Dict[str, Any]],
        retry_count: int,
        error: BaseException,
    ) -> EvaluationRecord:
        return EvaluationRecord(
            schema_version=self.schema_version,
            evaluator_model=self.model,
            prompt_version=self.prompt_version,
            timestamp=timestamp,
            source_generation_id=request.source_generation_id,
            dataset_id=request.dataset_id,
            evaluation_status=EvaluationStatus.FAILED,
            retry_count=retry_count,
            claims=[],
            generation_label=None,
            generation_failure_types=[],
            rationale="",
            needs_human_review=False,
            error_type=self._error_type_for(error),
            error_message=self._redact(str(error)),
            attempts=attempts,
        ).validate()

    # ------------------------------------------------------------------
    # Security helpers
    # ------------------------------------------------------------------
    @staticmethod
    def redact(text: str) -> str:
        """Replace any API-key-like token in ``text`` with a placeholder."""
        if not text:
            return text
        key = os.environ.get(API_KEY_ENV_VAR, "")
        if key:
            return text.replace(key, SECRET_PLACEHOLDER)
        return text

    @classmethod
    def api_key_from_environment(cls) -> Optional[str]:
        """Read the API key from the environment (never printed by callers)."""
        return os.environ.get(API_KEY_ENV_VAR)
