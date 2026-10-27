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
from typing import Any, Callable, Dict, Iterable, List, Optional, Protocol, Sequence, Tuple, Union

from .prompt import EVALUATOR_MODEL, PROMPT_VERSION, build_messages, response_schema
from .retry import (
    NonRetryableError,
    QuotaExhaustedError,
    RetryPolicy,
    RetryExhausted,
    RetryableError,
    retry_call,
)
from .schemas import (
    SCHEMA_VERSION,
    EvaluationRecord,
    EvaluationStatus,
    StructuredEvaluation,
)

API_KEY_ENV_VAR = "GEMINI_API_KEY"
MAX_CREDENTIAL_ATTEMPTS_ENV_VAR = "GEMINI_MAX_CREDENTIAL_ATTEMPTS"
SKIP_CREDENTIAL_SLOTS_ENV_VAR = "GEMINI_SKIP_CREDENTIAL_SLOTS"
SECRET_PLACEHOLDER = "<GEMINI_API_KEY-redacted>"


def discover_api_keys(
    api_key: Optional[Union[str, Sequence[str], Iterable[str]]] = None,
    environment: Optional[Dict[str, str]] = None,
) -> List[str]:
    """Return ordered Gemini API keys, preserving the primary key first.

    The project historically uses a single ``GEMINI_API_KEY``. Newer
    configurations may also include ``GEMINI_API_KEY_1``, ``_2``, etc., each
    backed by a different Google Cloud project. We preserve compatibility by
    accepting either a single string or a list/sequence of keys; when no
    explicit values are supplied, we read the environment in the canonical order
    ``GEMINI_API_KEY``, then ``GEMINI_API_KEY_1`` and up.
    """
    return [value for _, value in discover_api_credentials(api_key, environment)]


def discover_api_credentials(
    api_key: Optional[Union[str, Sequence[str], Iterable[str]]] = None,
    environment: Optional[Dict[str, str]] = None,
) -> List[Tuple[str, str]]:
    """Return ordered (environment slot, API key) pairs without exposing values."""
    env = os.environ if environment is None else environment
    credentials = []
    seen = set()

    def add(slot: str, value: Optional[str]) -> None:
        if value and value not in seen:
            credentials.append((slot, value))
            seen.add(value)

    if api_key is not None:
        if isinstance(api_key, str):
            add(API_KEY_ENV_VAR, api_key)
        else:
            for index, item in enumerate(api_key):
                add(API_KEY_ENV_VAR if index == 0 else f"{API_KEY_ENV_VAR}_{index}", str(item))
        return credentials

    numbered_slots = sorted(
        (
            name
            for name in env
            if name.startswith(f"{API_KEY_ENV_VAR}_")
            and name[len(API_KEY_ENV_VAR) + 1 :].isdigit()
        ),
        key=lambda name: int(name[len(API_KEY_ENV_VAR) + 1 :]),
    )
    for slot in [API_KEY_ENV_VAR] + numbered_slots:
        add(slot, env.get(slot))
    return credentials


class EvaluatorError(Exception):
    """Base evaluator error."""


class EvaluatorConfigError(EvaluatorError, NonRetryableError):
    """Configuration problem (missing key / SDK). Not retryable."""


class TransientEvaluationError(EvaluatorError, RetryableError):
    """Transient API failure (retryable)."""


class PersistentEvaluationError(EvaluatorError, NonRetryableError):
    """Persistent, non-retryable API failure (e.g. auth / invalid argument)."""


class CredentialAccessError(PersistentEvaluationError):
    """Credential/project-specific authentication or model access failure."""


class GlobalModelConfigurationError(PersistentEvaluationError):
    """Model deprecation or invalid request configuration shared by credentials."""


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
        api_key: Optional[Union[str, Sequence[str], Iterable[str]]] = None,
        client: Optional[Any] = None,
        retry_policy: Optional[RetryPolicy] = None,
        prompt_version: str = PROMPT_VERSION,
        schema_version: str = SCHEMA_VERSION,
        temperature: float = 0.2,
        max_credential_attempts: Optional[int] = None,
        excluded_credential_slots: Optional[Iterable[str]] = None,
        message_builder: Optional[
            Callable[[str, str, str], List[Dict[str, str]]]
        ] = None,
        response_schema_builder: Optional[Callable[[], Dict[str, Any]]] = None,
    ):
        self.model = model
        self.prompt_version = prompt_version
        self.schema_version = schema_version
        self.temperature = temperature
        self.message_builder = message_builder or build_messages
        self.response_schema_builder = response_schema_builder or response_schema
        self.retry_policy = retry_policy or RetryPolicy()
        self._injected_client = client
        self._cached_client: Optional[Any] = None
        self._all_api_credentials = discover_api_credentials(api_key=api_key)
        excluded_slots = {
            slot.strip()
            for slot in os.environ.get(SKIP_CREDENTIAL_SLOTS_ENV_VAR, "").split(",")
            if slot.strip()
        }
        excluded_slots.update(
            slot.strip() for slot in (excluded_credential_slots or ()) if slot.strip()
        )
        self._excluded_credential_slots = excluded_slots
        self._api_credentials = [
            (slot, key) for slot, key in self._all_api_credentials
            if slot not in excluded_slots
        ]
        self._api_key_candidates = [key for _, key in self._all_api_credentials]
        self._resolved_api_key = (
            self._api_credentials[0][1] if self._api_credentials else None
        )
        configured_max = max_credential_attempts
        if configured_max is None:
            raw_max = os.environ.get(MAX_CREDENTIAL_ATTEMPTS_ENV_VAR)
            if raw_max:
                try:
                    configured_max = int(raw_max)
                except ValueError as exc:
                    raise EvaluatorConfigError(
                        f"{MAX_CREDENTIAL_ATTEMPTS_ENV_VAR} must be a positive integer"
                    ) from exc
        if configured_max is not None and configured_max < 1:
            raise EvaluatorConfigError(
                f"{MAX_CREDENTIAL_ATTEMPTS_ENV_VAR} must be a positive integer"
            )
        self.max_credential_attempts = configured_max
        # The API key is deliberately never persisted or logged.

    def exclude_credential_slots(self, slots: Iterable[str]) -> None:
        """Exclude credentials already known unusable for the active model."""
        self._excluded_credential_slots.update(
            slot.strip() for slot in slots if slot.strip()
        )
        self._api_credentials = [
            (slot, key)
            for slot, key in self._all_api_credentials
            if slot not in self._excluded_credential_slots
        ]

    @staticmethod
    def _is_credential_specific_error(error: BaseException) -> bool:
        """Return whether another API key could fix an auth/model-access error."""
        if isinstance(error, CredentialAccessError):
            return True
        if isinstance(error, GlobalModelConfigurationError):
            return False
        root = error
        cause = getattr(error, "__cause__", None)
        while cause is not None:
            root = cause
            cause = getattr(cause, "__cause__", None)
        code = str(getattr(root, "code", ""))
        return code in ("401", "403", "404") or type(root).__name__ in (
            "Unauthorized",
            "PermissionDenied",
            "NotFound",
        )

    # ------------------------------------------------------------------
    # Client construction (lazy SDK import)
    # ------------------------------------------------------------------
    def _build_client(self) -> Any:
        """Create a real google.genai.Client. Lazy so tests need no SDK.

        A single instance is retained for the lifetime of the evaluator so
        that the underlying httpx transport is reused across attempts and
        is never closed mid-request. The caller is responsible for closing
        the client via :meth:`close` (or the context-manager protocol).
        """
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
        # httpx default timeout is None (no timeout), which means a request to
        # a high-demand model can hang indefinitely. Set an explicit ceiling so
        # transient stalls surface as a retryable TimeoutError instead of a
        # confusing RemoteProtocolError or an unbounded wait. 120 s is generous
        # for a structured-JSON evaluation call.
        http_options = genai.types.HttpOptions(timeout=120000)
        return genai.Client(api_key=self._resolved_api_key, http_options=http_options)

    def _client(self) -> Any:
        if self._injected_client is not None:
            return self._injected_client
        if self._cached_client is None:
            self._cached_client = self._build_client()
        return self._cached_client

    # ------------------------------------------------------------------
    # Client lifecycle
    # ------------------------------------------------------------------
    def close(self) -> None:
        """Close any SDK client we built (never an injected one)."""
        client = self._cached_client
        if client is not None and self._injected_client is None:
            close_fn = getattr(client, "close", None)
            if close_fn is not None:
                try:
                    close_fn()
                except Exception:  # noqa: BLE001 - best-effort cleanup
                    pass
        self._cached_client = None

    def __enter__(self) -> "GeminiEvaluator":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Request building
    # ------------------------------------------------------------------
    def _generation_config(self) -> Dict[str, Any]:
        return {
            "temperature": self.temperature,
            "response_mime_type": "application/json",
            "response_schema": self.response_schema_builder(),
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
        keys = list(self._api_key_candidates)
        if self._resolved_api_key and self._resolved_api_key not in keys:
            keys.append(self._resolved_api_key)
        for key in keys:
            if key:
                text = text.replace(key, SECRET_PLACEHOLDER)
        return text

    def _classify_api_error(self, exc: BaseException) -> EvaluatorError:
        if isinstance(exc, QuotaExhaustedError):
            return exc
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
            return CredentialAccessError(
                self._redact("gemini authentication/authorization failed: {0}".format(exc))
            )
        # Explicit deprecation/retirement and ordinary unknown-model/resource
        # responses are global configuration blockers, not a reason to try
        # further credentials.
        if code_str in ("400", "404") or name in ("InvalidArgument", "NotFound"):
            return GlobalModelConfigurationError(
                self._redact("gemini invalid argument/model configuration (not retried): {0}".format(exc))
            )
        # HTTP 429 with RESOURCE_EXHAUSTED (free-tier daily limit) is a
        # batch-stopping signal: retrying wastes API calls that will fail
        # until quota resets. Other 429s (per-minute rate-limit) are transient.
        if code_str == "429":
            status = getattr(exc, "status", "") or ""
            message = (str(exc) or "").lower()
            if (
                status == "RESOURCE_EXHAUSTED"
                or "resource_exhausted" in message
                or "quota" in message
                or "daily" in message
                or name == "ResourceExhausted"
            ):
                return QuotaExhaustedError(
                    self._redact(
                        "gemini quota exhausted (batch-stopping): {0}".format(exc)
                    )
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
        """Evaluate one generation with retry.

        Terminal ``evaluation_status`` is one of:
          * SUCCESS        - resolved; carries a generation_label.
          * MANUAL_REVIEW  - parsed but flagged uncertain.
          * FAILED         - transport/parse failure exhausted retries. NEVER
                               carries a generation_label and NEVER contributes
                               to empirical risk.

        When multiple Gemini API keys are configured, the evaluator tries the
        primary key first and falls back to additional eligible credentials only
        on confirmed quota exhaustion. This keeps the existing single-key
        behaviour while allowing a deterministic failover path for multi-project
        configurations.
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        messages = self.message_builder(
            request.question, request.context, request.generated_answer
        )
        attempts: List[Dict[str, Any]] = []
        policy = self.retry_policy
        credentials = list(self._api_credentials)

        if not credentials:
            if self._all_api_credentials and self._injected_client is None:
                raise EvaluatorConfigError(
                    "All configured Gemini credentials are excluded as previously unusable"
                )
            credentials = [(API_KEY_ENV_VAR, self._resolved_api_key or "")]
        max_attempts = self.max_credential_attempts
        if max_attempts is None:
            max_attempts = len(credentials)
        credentials = credentials[:max_attempts]

        for key_index, (current_key_label, api_key) in enumerate(credentials):
            self._resolved_api_key = api_key
            self._cached_client = None
            key_attempts: List[Dict[str, Any]] = []

            def _one_attempt() -> StructuredEvaluation:
                raw = self._call_gemini(messages)
                return self._parse(raw)

            def _log(attempt_number: int, exc: Optional[Exception]) -> None:
                if exc is None:
                    key_attempts.append({
                        "attempt": attempt_number,
                        "status": "SUCCESS",
                        "key_slot": current_key_label,
                    })
                else:
                    retryable = policy.is_retryable(exc) if policy else False
                    key_attempts.append(
                        {
                            "attempt": attempt_number,
                            "status": "RETRY" if retryable else "FAILED",
                            "error_type": type(exc).__name__,
                            "error_message": self._redact(str(exc)),
                            "key_slot": current_key_label,
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
                attempts.extend(key_attempts)
                if isinstance(exc.last_error, QuotaExhaustedError):
                    if (
                        key_index < len(credentials) - 1
                    ):
                        continue
                    exc.last_error.attempts = list(attempts)
                    raise exc.last_error
                retry_count = max(0, len(attempts) - 1)
                return self._failed_record(
                    request, timestamp, attempts, retry_count, exc.last_error
                )
            except PersistentEvaluationError as exc:
                attempts.extend(key_attempts)
                if (
                    self._is_credential_specific_error(exc)
                    and key_index < len(credentials) - 1
                ):
                    continue
                return self._failed_record(
                    request,
                    timestamp,
                    attempts,
                    retry_count=max(0, len(attempts) - 1),
                    error=exc,
                )
            except QuotaExhaustedError as exc:
                attempts.extend(key_attempts)
                if (
                    key_index < len(credentials) - 1
                ):
                    continue
                exc.attempts = list(attempts)
                raise

            attempts.extend(key_attempts)
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

        return self._failed_record(
            request,
            timestamp,
            attempts,
            retry_count=max(0, len(attempts) - 1),
            error=QuotaExhaustedError("gemini quota exhausted across all configured credentials"),
        )

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
