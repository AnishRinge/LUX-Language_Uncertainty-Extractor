"""Focused hermetic tests for the CP4.5-A smoke-test execution protocol.

These tests do NOT make live Gemini calls. They verify:
- the six representative generations are selected deterministically;
- the smoke-test protocol is quota-safe (max_retries=0, spaced calls) WITHOUT
  altering the evaluator's default retry policy;
- the harness builds requests that send only question/context/generated_answer;
- forbidden target-model metadata cannot enter the Gemini request;
- the CP4.2.5 source artifact is never a smoke output path.
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import run_cp4_5_a_smoke_test as smoke  # noqa: E402
from src.evaluation.gemini_evaluator import (  # noqa: E402
    API_KEY_ENV_VAR,
    GeminiEvaluator,
)
from src.evaluation.retry import RetryPolicy  # noqa: E402

ARTIFACT = Path("data/pilots/cp4_2_5_representative_validation.json")


def _json_response(**overrides):
    base = {
        "evaluation_status": "SUCCESS",
        "claims": [],
        "generation_label": "RELIABLE",
        "generation_failure_types": [],
        "rationale": "ok",
        "needs_human_review": False,
    }
    base.update(overrides)
    return json.dumps(base)


class _FakeResponse:
    def __init__(self, text):
        self.text = text


class _FakeModels:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def generate_content(self, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        entry = self.responses[min(len(self.calls) - 1, len(self.responses) - 1)]
        if isinstance(entry, BaseException):
            raise entry
        return _FakeResponse(entry)


class _FakeClient:
    def __init__(self, responses):
        self.models = _FakeModels(responses)


class TestSmokeProtocol(unittest.TestCase):
    def setUp(self):
        with open(ARTIFACT, "r", encoding="utf-8") as fh:
            self.artifact = json.load(fh)

    def test_selects_exactly_the_six_targets(self):
        selected = smoke.select_samples(self.artifact)
        indexes = [g["dataset_index"] for g in selected]
        self.assertEqual(sorted(indexes), sorted(smoke.TARGET_DATASET_INDEXES))
        self.assertEqual(len(selected), len(smoke.TARGET_DATASET_INDEXES))
        for g in selected:
            self.assertEqual(g["seed"], smoke.SEED)

    def test_smoke_retry_policy_is_zero(self):
        self.assertEqual(smoke.SMOKE_MAX_RETRIES, 0)
        self.assertEqual(smoke.SMOKE_RETRY_POLICY.max_retries, 0)

    def test_smoke_call_spacing_within_free_tier(self):
        # 12-15s keeps <= 5 requests/minute on the free tier.
        self.assertGreaterEqual(smoke.SMOKE_CALL_SPACING_SECONDS, 12)
        self.assertLessEqual(smoke.SMOKE_CALL_SPACING_SECONDS, 15)

    def test_normal_evaluator_default_retry_policy_untouched(self):
        # The evaluator's default must remain the standard retry policy.
        default = RetryPolicy()
        self.assertEqual(default.max_retries, 3)
        ev = GeminiEvaluator(client=_FakeClient([_json_response()]))
        self.assertEqual(ev.retry_policy.max_retries, 3)
        # The smoke policy is a distinct, zero-retry instance.
        self.assertEqual(smoke.SMOKE_RETRY_POLICY.max_retries, 0)
        self.assertNotEqual(id(ev.retry_policy), id(smoke.SMOKE_RETRY_POLICY))


class TestSmokeRequestIsolation(unittest.TestCase):
    def setUp(self):
        with open(ARTIFACT, "r", encoding="utf-8") as fh:
            self.artifact = json.load(fh)

    def test_build_request_only_exposes_allowed_fields(self):
        selected = smoke.select_samples(self.artifact)
        req = smoke.build_request(selected[0])
        # Forbidden target-model identity must be blank on the request object.
        self.assertEqual(req.model_name, "")
        self.assertEqual(req.model_revision, "")
        # question/context/generated_answer are the evaluator inputs.
        self.assertTrue(req.question.strip().endswith("?"))
        self.assertTrue(req.context.strip())
        self.assertTrue(req.generated_answer.strip())

    def test_payload_sent_to_gemini_is_question_context_answer_only(self):
        selected = smoke.select_samples(self.artifact)
        req = smoke.build_request(selected[0])
        client = _FakeClient([_json_response()])
        ev = GeminiEvaluator(client=client, retry_policy=smoke.SMOKE_RETRY_POLICY)
        ev.evaluate(req)

        call = client.models.calls[0]
        contents = call["contents"]
        # Must NOT be OpenAI-style message dicts.
        self.assertNotIsInstance(contents, list)
        self.assertIsInstance(contents, str)
        # Only the three allowed evidentiary inputs appear.
        self.assertIn(req.question, contents)
        self.assertIn(req.context, contents)
        self.assertIn(req.generated_answer, contents)
        # Forbidden target-model features do not appear in the wire payload.
        blob = json.dumps(call, default=str)
        self.assertNotIn("FORBIDDEN", blob)
        # model_name and model_revision are intentionally blank on the request
        # (no Qwen3 identity is ever sent to the Gemini evaluator).
        self.assertEqual(req.model_name, "")
        self.assertEqual(req.model_revision, "")
        # The seed (provenance only) must not appear in the Gemini user turn.
        self.assertNotIn(str(req.seed), contents)


class TestSmokeArtifactSafety(unittest.TestCase):
    def test_source_artifact_is_not_a_smoke_output(self):
        # The harness must never write back to the CP4.2.5 source artifact.
        self.assertTrue(smoke.ARTIFACT.exists())
        self.assertNotIn(
            smoke.ARTIFACT,
            [smoke.SMOKE_OUTPUTS["artifact"], smoke.SMOKE_OUTPUTS["report"]],
        )


class TestSmokeRunExecution(unittest.TestCase):
    """Verify run() makes exactly one attempt per case with zero retries and
    ~15s spacing, WITHOUT altering the evaluator's normal retry policy.

    These tests inject a fake client via _build_client patching — no SDK,
    no network, no API key required for the call itself (the env-var gate
    is satisfied with a placeholder).
    """

    def setUp(self):
        with open(ARTIFACT, "r", encoding="utf-8") as fh:
            self.artifact = json.load(fh)

    def _run_with_fake(self, fake_client):
        """Execute smoke.run() with a fake Gemini client and temp outputs.

        Returns (mock_sleep, artifact_path) for assertion. The temp directory
        is cleaned up after the test via addCleanup, not a context manager, so
        the written artifact survives for post-run assertions.
        """
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        artifact_path = Path(tmpdir.name) / "smoke_artifact.json"
        report_path = Path(tmpdir.name) / "smoke_report.md"
        with patch.object(smoke, "SMOKE_OUTPUTS", {
            "artifact": artifact_path,
            "report": report_path,
        }):
            with patch.object(smoke.time, "sleep") as mock_sleep:
                with patch.object(
                    GeminiEvaluator, "_build_client",
                    return_value=fake_client,
                ):
                    with patch.dict(os.environ, {API_KEY_ENV_VAR: "test-key"}):
                        smoke.run()
        return mock_sleep, artifact_path

    def test_run_makes_exactly_one_call_per_case(self):
        fake_client = _FakeClient([_json_response() for _ in range(6)])
        mock_sleep, artifact_path = self._run_with_fake(fake_client)
        # Exactly 6 Gemini calls = one per case (zero retries).
        self.assertEqual(len(fake_client.models.calls), 6)
        # 5 inter-case sleeps at the smoke spacing (15s).
        self.assertEqual(mock_sleep.call_count, 5)
        for call in mock_sleep.call_args_list:
            self.assertEqual(call.args[0], smoke.SMOKE_CALL_SPACING_SECONDS)
        # Output artifact has 6 records, 0 retries.
        data = json.loads(artifact_path.read_text(encoding="utf-8"))
        self.assertEqual(len(data["records"]), 6)
        self.assertEqual(data["summary"]["retries"], 0)

    def test_run_does_not_retry_on_failure(self):
        # All responses are retryable failures; with max_retries=0 each case
        # gets exactly one call and zero retries. If retries were >0 we would
        # see more than 6 calls here.
        fake_client = _FakeClient([TimeoutError("transient")] * 6)
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_path = Path(tmpdir) / "smoke_artifact.json"
            report_path = Path(tmpdir) / "smoke_report.md"
            with patch.object(smoke, "SMOKE_OUTPUTS", {
                "artifact": artifact_path,
                "report": report_path,
            }):
                with patch.object(smoke.time, "sleep"):
                    with patch.object(
                        GeminiEvaluator, "_build_client",
                        return_value=fake_client,
                    ):
                        with patch.dict(os.environ, {API_KEY_ENV_VAR: "test-key"}):
                            smoke.run()

            data = json.loads(artifact_path.read_text(encoding="utf-8"))
            # All-failing responses, but still only 6 calls (no retries).
            self.assertEqual(len(fake_client.models.calls), 6)
            self.assertEqual(len(data["records"]), 6)
            self.assertEqual(data["summary"]["retries"], 0)
            for r in data["records"]:
                self.assertEqual(r["evaluation_status"], "FAILED")
                self.assertEqual(r["retry_count"], 0)

    def test_run_records_smoke_retry_config(self):
        fake_client = _FakeClient([_json_response() for _ in range(6)])
        mock_sleep, artifact_path = self._run_with_fake(fake_client)
        data = json.loads(artifact_path.read_text(encoding="utf-8"))
        self.assertEqual(
            data["run_metadata"]["smoke_retry_max_retries"],
            smoke.SMOKE_MAX_RETRIES,
        )
        self.assertEqual(
            data["run_metadata"]["smoke_call_spacing_seconds"],
            smoke.SMOKE_CALL_SPACING_SECONDS,
        )
        self.assertEqual(
            data["run_metadata"]["selected_dataset_indexes"],
            smoke.TARGET_DATASET_INDEXES,
        )

    def test_run_does_not_alter_evaluator_default_policy(self):
        # Confirm the default RetryPolicy still has max_retries=3 after
        # the smoke module has been imported.
        default = RetryPolicy()
        self.assertEqual(default.max_retries, 3)
        self.assertEqual(smoke.SMOKE_RETRY_POLICY.max_retries, 0)
        self.assertNotEqual(default.max_retries, smoke.SMOKE_RETRY_POLICY.max_retries)


if __name__ == "__main__":
    unittest.main()
