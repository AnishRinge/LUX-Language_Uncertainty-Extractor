"""Tests for the CP4.4 Gemini evaluator infrastructure.

All tests are hermetic: they inject a fake Gemini client (no SDK, no network,
no API key required). They cover valid structured output, API failures, retry
behaviour, persistent failure -> FAILED, parsing failure -> retry, semantic
uncertainty -> MANUAL_REVIEW, the FAILED-never-becomes-a-label invariant,
API-key-from-environment, and API-key-never-logged.
"""

import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.gemini_evaluator import (
    API_KEY_ENV_VAR,
    EvaluationParseError,
    EvaluationRequest,
    GeminiEvaluator,
    PersistentEvaluationError,
    TransientEvaluationError,
)
from src.evaluation.prompt import EVALUATOR_SYSTEM_INSTRUCTION
from src.evaluation.retry import RetryPolicy
from src.evaluation.schemas import (
    EvaluationRecord,
    EvaluationStatus,
    FailureType,
    GenerationLabel,
)


SECRET_KEY = "super-secret-test-key-12345"


def _request(answer="Some answer.", question="Q?", context="C."):
    return EvaluationRequest(
        question=question,
        context=context,
        generated_answer=answer,
        source_generation_id="squad_v2_test:0:1001",
        dataset_id="squad_v2",
        sample_id="squad_v2_test",
        generation_index=1,
        seed=1001,
    )


def _json_response(**overrides):
    base = {
        "evaluation_status": "SUCCESS",
        "claims": [
            {
                "claim_text": "The answer states X.",
                "claim_status": "SUPPORTED",
                "evidence_text": "context says X",
                "failure_type": "NONE",
                "materiality": "NON_MATERIAL",
            }
        ],
        "generation_label": "RELIABLE",
        "generation_failure_types": [],
        "rationale": "evidence supports the claim",
        "needs_human_review": False,
    }
    base.update(overrides)
    return json.dumps(base)


class _FakeResponse:
    def __init__(self, text):
        self.text = text


class _FakeModels:
    """Configurable fake: ``responses`` is a list of JSON strings or Exceptions.

    The last entry is reused for any extra calls.
    """

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def generate_content(self, model, contents, config):
        self.calls.append(
            {"model": model, "contents": contents, "config": config}
        )
        i = min(len(self.calls) - 1, len(self.responses) - 1)
        entry = self.responses[i]
        if isinstance(entry, BaseException):
            raise entry
        return _FakeResponse(entry)


class _FakeClient:
    def __init__(self, responses):
        self.models = _FakeModels(responses)


class _AuthError(Exception):
    code = 401


# --------------------------------------------------------------------------
# 1. valid structured evaluator output (end-to-end through the evaluator)
# --------------------------------------------------------------------------
class TestValidOutput(unittest.TestCase):
    def test_valid_response_yields_success_label(self):
        client = _FakeClient([_json_response(generation_label="RELIABLE")])
        ev = GeminiEvaluator(model="gemini-3.8-flash", client=client,
                             retry_policy=RetryPolicy(max_retries=0))
        record = ev.evaluate(_request())
        self.assertEqual(record.evaluation_status, EvaluationStatus.SUCCESS)
        self.assertEqual(record.generation_label, GenerationLabel.RELIABLE)
        self.assertEqual(len(client.models.calls), 1)
        self.assertEqual(record.retry_count, 0)

    def test_unreliable_response_with_material_contradiction(self):
        resp = _json_response(
            evaluation_status="SUCCESS",
            claims=[
                {"claim_text": "Paris is the capital.", "claim_status": "CONTRADICTED",
                 "evidence_text": "context says London", "failure_type": "CONTEXT_CONTRADICTION",
                 "materiality": "MATERIAL"},
            ],
            generation_label="UNRELIABLE",
            generation_failure_types=["CONTEXT_CONTRADICTION"],
            rationale="contradicts context",
            needs_human_review=False,
        )
        client = _FakeClient([resp])
        ev = GeminiEvaluator(client=client, retry_policy=RetryPolicy(max_retries=0))
        record = ev.evaluate(_request())
        self.assertEqual(record.evaluation_status, EvaluationStatus.SUCCESS)
        self.assertEqual(record.generation_label, GenerationLabel.UNRELIABLE)
        self.assertIn("CONTEXT_CONTRADICTION", record.generation_failure_types)


# --------------------------------------------------------------------------
# 4. evaluator API failure  /  6. persistent failure -> FAILED
# --------------------------------------------------------------------------
class TestApiFailure(unittest.TestCase):
    def test_persistent_auth_failure_yields_failed(self):
        client = _FakeClient([_AuthError("unauthorized")])
        ev = GeminiEvaluator(
            client=client,
            retry_policy=RetryPolicy(max_retries=3, sleep=lambda s: None),
        )
        record = ev.evaluate(_request())
        self.assertEqual(record.evaluation_status, EvaluationStatus.FAILED)
        self.assertIsNone(record.generation_label)
        self.assertEqual(record.error_type, "PersistentEvaluationError")
        self.assertEqual(record.retry_count, 0)  # not retried
        self.assertEqual(len(client.models.calls), 1)

    def test_retryable_exhaustion_yields_failed(self):
        # Timeout on every attempt -> retried up to max_retries, then FAILED.
        client = _FakeClient([TimeoutError("deadline exceeded")] * 5)
        ev = GeminiEvaluator(
            client=client,
            retry_policy=RetryPolicy(max_retries=2, sleep=lambda s: None),
        )
        record = ev.evaluate(_request())
        self.assertEqual(record.evaluation_status, EvaluationStatus.FAILED)
        self.assertIsNone(record.generation_label)
        self.assertIn("TimeoutError", record.error_type)
        # 1 initial attempt + 2 retries = 3 calls.
        self.assertEqual(len(client.models.calls), 3)
        self.assertEqual(record.retry_count, 2)
        self.assertEqual(len(record.attempts), 3)


# --------------------------------------------------------------------------
# 5. retry behaviour
# --------------------------------------------------------------------------
class TestRetry(unittest.TestCase):
    def test_transient_then_success(self):
        client = _FakeClient(
            [TransientEvaluationError("rate limited"), _json_response()]
        )
        ev = GeminiEvaluator(
            client=client,
            retry_policy=RetryPolicy(max_retries=3, sleep=lambda s: None),
        )
        record = ev.evaluate(_request())
        self.assertEqual(record.evaluation_status, EvaluationStatus.SUCCESS)
        self.assertEqual(record.retry_count, 1)
        self.assertEqual(len(client.models.calls), 2)
        self.assertEqual(record.attempts[0]["status"], "RETRY")
        self.assertEqual(record.attempts[-1]["status"], "SUCCESS")

    def test_retry_delay_is_configurable_not_hardcoded(self):
        policy = RetryPolicy(max_retries=5, base_delay=2.0, backoff_factor=3.0, sleep=lambda s: None)
        self.assertEqual(policy.delay_for(1), 2.0)
        self.assertEqual(policy.delay_for(2), 6.0)
        self.assertLessEqual(policy.delay_for(5), policy.max_delay)


# --------------------------------------------------------------------------
# 7. parsing failure -> retry
# --------------------------------------------------------------------------
class TestParseFailure(unittest.TestCase):
    def test_bad_json_then_valid(self):
        client = _FakeClient(["not valid json at all", _json_response()])
        ev = GeminiEvaluator(
            client=client,
            retry_policy=RetryPolicy(max_retries=3, sleep=lambda s: None),
        )
        record = ev.evaluate(_request())
        self.assertEqual(record.evaluation_status, EvaluationStatus.SUCCESS)
        self.assertEqual(record.retry_count, 1)
        self.assertEqual(len(record.attempts), 2)
        # The parse failure was logged as a retry.
        self.assertEqual(record.attempts[0]["status"], "RETRY")

    def test_schema_violation_then_valid(self):
        bad = _json_response(generation_label="MAYBE")  # invalid enum
        client = _FakeClient([bad, _json_response()])
        ev = GeminiEvaluator(
            client=client,
            retry_policy=RetryPolicy(max_retries=3, sleep=lambda s: None),
        )
        record = ev.evaluate(_request())
        self.assertEqual(record.evaluation_status, EvaluationStatus.SUCCESS)
        self.assertEqual(record.retry_count, 1)

    def test_fence_stripped(self):
        fenced = "```json\n" + _json_response() + "\n```"
        client = _FakeClient([fenced])
        ev = GeminiEvaluator(client=client, retry_policy=RetryPolicy(max_retries=0))
        record = ev.evaluate(_request())
        self.assertEqual(record.evaluation_status, EvaluationStatus.SUCCESS)

    def test_parse_exhaustion_yields_failed(self):
        client = _FakeClient(["not json"] * 4)
        ev = GeminiEvaluator(
            client=client,
            retry_policy=RetryPolicy(max_retries=2, sleep=lambda s: None),
        )
        record = ev.evaluate(_request())
        self.assertEqual(record.evaluation_status, EvaluationStatus.FAILED)
        self.assertIsNone(record.generation_label)
        self.assertIn("Parse", record.error_type)


# --------------------------------------------------------------------------
# 8. semantic uncertainty -> MANUAL_REVIEW
# --------------------------------------------------------------------------
class TestSemanticUncertainty(unittest.TestCase):
    def test_needs_human_review_flag_promotes_to_manual_review(self):
        resp = _json_response(
            generation_label="INADEQUATE",
            claims=[
                {"claim_text": "unclear", "claim_status": "UNVERIFIABLE",
                 "evidence_text": "cannot determine",
                 "failure_type": "AMBIGUOUS_INTERPRETATION",
                 "materiality": "UNRESOLVED"},
            ],
            generation_failure_types=[],
            rationale="cannot confidently resolve",
            needs_human_review=True,
        )
        client = _FakeClient([resp])
        ev = GeminiEvaluator(client=client, retry_policy=RetryPolicy(max_retries=0))
        record = ev.evaluate(_request())
        self.assertEqual(record.evaluation_status, EvaluationStatus.MANUAL_REVIEW)
        self.assertTrue(record.needs_human_review)
        # Still carries the label the model produced (reserved, not in risk).
        self.assertEqual(record.generation_label, GenerationLabel.INADEQUATE)

    def test_model_signals_manual_review(self):
        resp = _json_response(
            evaluation_status="MANUAL_REVIEW",
            needs_human_review=True,
        )
        client = _FakeClient([resp])
        ev = GeminiEvaluator(client=client, retry_policy=RetryPolicy(max_retries=0))
        record = ev.evaluate(_request())
        self.assertEqual(record.evaluation_status, EvaluationStatus.MANUAL_REVIEW)


# --------------------------------------------------------------------------
# 9. FAILED never becomes a generation label
# --------------------------------------------------------------------------
class TestFailedNeverBecomesLabel(unittest.TestCase):
    def test_failed_record_has_no_label_and_no_claims(self):
        client = _FakeClient([_AuthError("401")])
        ev = GeminiEvaluator(client=client, retry_policy=RetryPolicy(max_retries=3, sleep=lambda s: None))
        record = ev.evaluate(_request())
        self.assertEqual(record.evaluation_status, EvaluationStatus.FAILED)
        self.assertIsNone(record.generation_label)
        self.assertEqual(record.claims, [])
        # Never one of the generation labels.
        self.assertNotIn(record.evaluation_status, GenerationLabel.__members__.values())

    def test_failed_record_serialization_keeps_label_null(self):
        client = _FakeClient([_AuthError("forbidden")])
        ev = GeminiEvaluator(client=client, retry_policy=RetryPolicy(max_retries=1, sleep=lambda s: None))
        record = ev.evaluate(_request())
        d = record.to_dict()
        self.assertIsNone(d["generation_label"])
        self.assertIsNone(d["claims"])
        dumped = json.dumps(d)
        self.assertNotIn("RELIABLE", dumped)
        self.assertNotIn("UNRELIABLE", dumped)
        self.assertNotIn("INADEQUATE", dumped)


# --------------------------------------------------------------------------
# 10. UNVERIFIABLE != UNSUPPORTED at the evaluator level
# --------------------------------------------------------------------------
class TestUnverifiableVsUnsupportedEvaluator(unittest.TestCase):
    def test_unverifiable_non_material_claim_keeps_reliable(self):
        resp = _json_response(
            claims=[
                {"claim_text": "main answer", "claim_status": "SUPPORTED",
                 "evidence_text": "ctx", "failure_type": "NONE",
                 "materiality": "NON_MATERIAL"},
                {"claim_text": "peripheral detail", "claim_status": "UNVERIFIABLE",
                 "evidence_text": "not determinable", "failure_type": "AMBIGUOUS_INTERPRETATION",
                 "materiality": "UNRESOLVED"},
            ],
            needs_human_review=True,
            generation_label="INADEQUATE",
        )
        client = _FakeClient([resp])
        ev = GeminiEvaluator(client=client, retry_policy=RetryPolicy(max_retries=0))
        record = ev.evaluate(_request())
        # The UNVERIFIABLE claim did not flip the model's conservative label;
        # the model signalled uncertainty -> MANUAL_REVIEW, not UNRELIABLE.
        self.assertEqual(record.evaluation_status, EvaluationStatus.MANUAL_REVIEW)
        statuses = [c.claim_status.value for c in record.claims]
        self.assertIn("UNVERIFIABLE", statuses)
        self.assertNotIn("UNSUPPORTED", statuses)


# --------------------------------------------------------------------------
# 13. API key is loaded from environment
# 14. API key is never logged
# --------------------------------------------------------------------------
class TestApiKeyHandling(unittest.TestCase):
    def test_api_key_loaded_from_environment(self):
        with patch.dict(os.environ, {API_KEY_ENV_VAR: SECRET_KEY}):
            ev = GeminiEvaluator()  # no explicit key, no injected client
            self.assertEqual(ev._resolved_api_key, SECRET_KEY)

    def test_api_key_not_required_when_client_injected(self):
        env = {API_KEY_ENV_VAR: SECRET_KEY}
        # Even without the env var set, an injected client must work.
        ev = GeminiEvaluator(client=_FakeClient([_json_response()]))
        self.assertIsNone(ev._resolved_api_key)
        record = ev.evaluate(_request())
        self.assertEqual(record.evaluation_status, EvaluationStatus.SUCCESS)

    def test_api_key_never_appears_in_failed_record(self):
        # Fake client raises an exception whose message contains the key; the
        # key must be redacted before any persistence.
        with patch.dict(os.environ, {API_KEY_ENV_VAR: SECRET_KEY}):

            class _Leak(Exception):
                pass

            err = _Leak("request to https://gemini?key=" + SECRET_KEY + " failed")
            client = _FakeClient([err])
            ev = GeminiEvaluator(
                client=client,
                retry_policy=RetryPolicy(max_retries=2, sleep=lambda s: None),
            )
            record = ev.evaluate(_request())
        self.assertEqual(record.evaluation_status, EvaluationStatus.FAILED)
        dumped = json.dumps(record.to_dict())
        self.assertNotIn(SECRET_KEY, dumped)
        self.assertIn("redacted", dumped.lower())

    def test_api_key_never_sent_in_prompt(self):
        with patch.dict(os.environ, {API_KEY_ENV_VAR: SECRET_KEY}):
            client = _FakeClient([_json_response()])
            ev = GeminiEvaluator(client=client, retry_policy=RetryPolicy(max_retries=0))
            ev.evaluate(_request())
        sent = json.dumps(client.models.calls[0])
        self.assertNotIn(SECRET_KEY, sent)

    def test_redact_helper(self):
        with patch.dict(os.environ, {API_KEY_ENV_VAR: SECRET_KEY}):
            ev = GeminiEvaluator()
        out = ev._redact("token=" + SECRET_KEY + " here")
        self.assertNotIn(SECRET_KEY, out)
        self.assertIn("<GEMINI_API_KEY-redacted>", out)

    def test_missing_api_key_and_sdk_raises_config_error(self):
        # No env key, no injected client, SDK absent -> config error, not a key.
        env = {k: v for k, v in os.environ.items() if k != API_KEY_ENV_VAR}
        with patch.dict(os.environ, env, clear=True):
            ev = GeminiEvaluator()
            with self.assertRaises(Exception):
                ev.evaluate(_request())


# --------------------------------------------------------------------------
# 11. SDK request compatibility and input isolation (CP4.5-A fix)
# --------------------------------------------------------------------------
class TestRequestFormat(unittest.TestCase):
    def test_contents_is_sdk_compatible_string_not_openai_dicts(self):
        client = _FakeClient([_json_response()])
        ev = GeminiEvaluator(client=client, retry_policy=RetryPolicy(max_retries=0))
        req = _request(
            question="What is the acronym for the accrediting body?",
            context="The KU School of Engineering is ABET accredited.",
            answer="The acronym is ABET.",
        )
        ev.evaluate(req)
        call = client.models.calls[0]
        contents = call["contents"]
        # Must NOT be OpenAI-style {role, content} message dicts.
        self.assertNotIsInstance(contents, list)
        self.assertIsInstance(contents, str)
        # Only allowed evidence enters the user turn.
        self.assertIn("acronym for the accrediting body", contents)
        self.assertIn("ABET accredited", contents)
        self.assertIn("The acronym is ABET.", contents)
        # System instruction lives in config.system_instruction, not as a message.
        self.assertEqual(call["config"]["system_instruction"], EVALUATOR_SYSTEM_INSTRUCTION)
        self.assertNotIn("messages", call["config"])

    def test_forbidden_target_model_metadata_is_not_sent(self):
        client = _FakeClient([_json_response()])
        ev = GeminiEvaluator(client=client, retry_policy=RetryPolicy(max_retries=0))
        req = EvaluationRequest(
            question="Q?",
            context="C.",
            generated_answer="A.",
            source_generation_id="squad_v2_test:0:1001",
            dataset_id="squad_v2",
            sample_id="squad_v2_test",
            generation_index=1,
            seed=777777,
            model_name="FORBIDDEN_QWEN3_IDENTITY_TOKEN",
            model_revision="deadbeefcafebabe",
            answerability="ANSWERABLE",
        )
        ev.evaluate(req)
        blob = json.dumps(client.models.calls[0], default=str)
        # These predictor/provenance features must not reach the Gemini request.
        self.assertNotIn("FORBIDDEN_QWEN3_IDENTITY_TOKEN", blob)
        self.assertNotIn("deadbeefcafebabe", blob)
        self.assertNotIn("777777", blob)

    def test_deterministic_sdk_validation_error_is_not_retried(self):
        # A pydantic-style SDK request-validation error is deterministic; it must
        # surface as a persistent failure and must NOT be retried.
        class ValidationError(Exception):
            pass

        err = ValidationError(
            "12 validation errors for _GenerateContentParameters\n"
            "  contents.str\n    Input should be a valid string"
        )
        client = _FakeClient([err])
        ev = GeminiEvaluator(
            client=client, retry_policy=RetryPolicy(max_retries=3, sleep=lambda s: None)
        )
        record = ev.evaluate(_request())
        self.assertEqual(record.evaluation_status, EvaluationStatus.FAILED)
        self.assertEqual(len(client.models.calls), 1)  # not retried
        self.assertEqual(record.retry_count, 0)
        self.assertIn("validation", record.error_message.lower())

    def test_genuine_transient_error_is_still_retried(self):
        # Regression guard: transient failures keep using the retry policy.
        client = _FakeClient(
            [TransientEvaluationError("rate limited"), _json_response()]
        )
        ev = GeminiEvaluator(
            client=client, retry_policy=RetryPolicy(max_retries=3, sleep=lambda s: None)
        )
        record = ev.evaluate(_request())
        self.assertEqual(record.evaluation_status, EvaluationStatus.SUCCESS)
        self.assertEqual(record.retry_count, 1)
        self.assertEqual(len(client.models.calls), 2)


if __name__ == "__main__":
    unittest.main()


# --------------------------------------------------------------------------
# 12. Gemini wire-format compatibility (CP4.5-A 400 fix)
# --------------------------------------------------------------------------
def _walk_forbidden(obj, hits):
    """Recursively record forbidden wire-format keys/values found in a schema."""
    if isinstance(obj, dict):
        if "additionalProperties" in obj:
            hits.append("additionalProperties")
        for k, v in obj.items():
            if k == "response_modalities" and "JSON" in (v if isinstance(v, list) else [v]):
                hits.append("response_modalities_json")
            _walk_forbidden(v, hits)
    elif isinstance(obj, list):
        for v in obj:
            _walk_forbidden(v, hits)


class TestWireFormat(unittest.TestCase):
    def _single_call(self, responses):
        client = _FakeClient(responses)
        ev = GeminiEvaluator(client=client, retry_policy=RetryPolicy(max_retries=0))
        ev.evaluate(_request())
        return client.models.calls[0]

    def test_response_schema_has_no_additional_properties(self):
        call = self._single_call([_json_response()])
        schema = call["config"]["response_schema"]
        hits = []
        _walk_forbidden(schema, hits)
        self.assertEqual(hits, [], "response_schema must not contain additionalProperties")

    def test_response_config_uses_application_json(self):
        call = self._single_call([_json_response()])
        cfg = call["config"]
        self.assertEqual(cfg.get("response_mime_type"), "application/json")
        self.assertNotIn("response_modalities", cfg)

    def test_response_modalities_never_contains_json(self):
        # Regardless of how the config is built, "JSON" must never be a modality.
        call = self._single_call([_json_response()])
        cfg = call["config"]
        modalities = cfg.get("response_modalities")
        if modalities is not None:
            self.assertNotIn("JSON", modalities)

    def test_request_contains_only_allowed_inputs(self):
        # Forbidden target-model/provenance features must not reach Gemini.
        client = _FakeClient([_json_response()])
        ev = GeminiEvaluator(client=client, retry_policy=RetryPolicy(max_retries=0))
        req = _request(
            question="What is the acronym for the accrediting body?",
            context="ABET accredits engineering schools.",
            answer="The acronym is ABET.",
        )
        req.seed = 777777
        req.model_name = "FORBIDDEN_QWEN3_IDENTITY_TOKEN"
        req.model_revision = "deadbeefcafebabe"
        ev.evaluate(req)
        blob = json.dumps(client.models.calls[0], default=str)
        self.assertIn("acronym for the accrediting body", blob)
        self.assertIn("ABET accredits engineering schools.", blob)
        self.assertIn("The acronym is ABET.", blob)
        self.assertNotIn("FORBIDDEN_QWEN3_IDENTITY_TOKEN", blob)
        self.assertNotIn("deadbeefcafebabe", blob)
        self.assertNotIn("777777", blob)

    def test_http_400_invalid_argument_is_not_retried(self):
        class _InvalidArgument(Exception):
            code = 400

        client = _FakeClient([_InvalidArgument("INVALID_ARGUMENT: bad schema")])
        ev = GeminiEvaluator(
            client=client, retry_policy=RetryPolicy(max_retries=3, sleep=lambda s: None)
        )
        record = ev.evaluate(_request())
        self.assertEqual(record.evaluation_status, EvaluationStatus.FAILED)
        self.assertEqual(len(client.models.calls), 1)  # deterministic -> not retried
        self.assertEqual(record.retry_count, 0)


if __name__ == "__main__":
    unittest.main()

