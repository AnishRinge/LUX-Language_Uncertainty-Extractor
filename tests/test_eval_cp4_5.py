import json
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import run_eval_cp4_5
import run_eval_cp4_6
from src.evaluation.schemas import (
    EvaluationRecord,
    EvaluationStatus,
    GenerationLabel,
    SCHEMA_VERSION,
)
from src.evaluation.prompt import EVALUATOR_SYSTEM_INSTRUCTION
from src.evaluation.prompt_cp4_5 import build_messages, response_schema
from run_predictor import _bootstrap_metric_intervals


class _FakeEvaluator:
    calls = []
    requests = []

    def __init__(
        self,
        model,
        prompt_version,
        excluded_credential_slots=(),
        message_builder=None,
        response_schema_builder=None,
    ):
        self.model = model
        self.prompt_version = prompt_version
        self.excluded = set(excluded_credential_slots)

    def evaluate(self, request):
        self.calls.append(request.source_generation_id)
        self.requests.append(request)
        return EvaluationRecord(
            schema_version=SCHEMA_VERSION,
            evaluator_model=self.model,
            prompt_version=self.prompt_version,
            timestamp="2026-10-17T00:00:00+00:00",
            source_generation_id=request.source_generation_id,
            dataset_id=request.dataset_id,
            evaluation_status=EvaluationStatus.SUCCESS,
            retry_count=0,
            generation_label=GenerationLabel.RELIABLE,
        ).validate()

    def exclude_credential_slots(self, slots):
        self.excluded.update(slots)

    def close(self):
        return None


class TestCP45Evaluation(unittest.TestCase):
    def setUp(self):
        _FakeEvaluator.calls = []
        _FakeEvaluator.requests = []
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.manifest = self.root / "generations.json"
        self.scenarios_path = self.root / "scenarios.json"
        self.evaluations = self.root / "new-version-evaluations"
        self.log = self.root / "run.json"
        self.scenario = {
            "source": {"dataset": "squad_v2"},
            "prompt": {"question": "not the stored question", "context": "not stored"},
        }
        self.generations = [
            {
                "generation_id": f"sample:{i}:1001",
                "sample_id": "sample",
                "dataset_index": i,
                "seed": 1001,
                "prompt_id": "sample",
                "generation_index": 0,
                "generation_status": "SUCCESS",
                "model_name": "Qwen/Qwen3-1.7B",
                "model_revision": run_eval_cp4_5.MODEL_REVISION,
                "generated_answer": "answer",
                "question": "Q?",
                "context": "evidence",
            }
            for i in [0]
        ]
        self._write_inputs()

    def tearDown(self):
        self.temp.cleanup()

    def _write_inputs(self):
        self.manifest.write_text(
            json.dumps({
                "model": {
                    "name": "Qwen/Qwen3-1.7B",
                    "revision": run_eval_cp4_5.MODEL_REVISION,
                },
                "generations": self.generations,
            }),
            encoding="utf-8",
        )
        self.scenarios_path.write_text(
            json.dumps([self.scenario]), encoding="utf-8"
        )

    def test_versioned_prompt_preserves_instructions_and_three_labels(self):
        messages = build_messages("q", "context", "answer")
        schema_labels = response_schema()["properties"]["generation_label"]["enum"]
        self.assertEqual(EVALUATOR_SYSTEM_INSTRUCTION, messages[0]["content"])
        self.assertEqual(
            ["RELIABLE", "UNRELIABLE", "INADEQUATE"],
            schema_labels,
        )

    def test_evaluates_existing_generation_into_isolated_version_and_resumes(self):
        with patch.object(run_eval_cp4_5, "discover_api_credentials", return_value=[]):
            result = run_eval_cp4_5.run_evaluation(
                self.manifest,
                self.scenarios_path,
                self.evaluations,
                self.log,
                spacing_seconds=0,
                evaluator_class=_FakeEvaluator,
            )
            resumed = run_eval_cp4_5.run_evaluation(
                self.manifest,
                self.scenarios_path,
                self.evaluations,
                self.log,
                spacing_seconds=0,
                evaluator_class=_FakeEvaluator,
            )

        self.assertEqual(["sample:0:1001"], _FakeEvaluator.calls)
        self.assertEqual("Q?", _FakeEvaluator.requests[0].question)
        self.assertEqual("evidence", _FakeEvaluator.requests[0].context)
        record_path = self.evaluations / "sample_0_1001.json"
        record = json.loads(record_path.read_text(encoding="utf-8"))
        self.assertEqual("gemini-3.8-flash", record["evaluator_model"])
        self.assertEqual("CP4.5.0", record["prompt_version"])
        self.assertEqual("completed", result["stop_reason"])
        self.assertEqual("completed", resumed["stop_reason"])
        self.assertEqual(1, resumed["completed_record_count"])

    def test_run_log_rejects_a_different_generation_cohort(self):
        with patch.object(run_eval_cp4_5, "discover_api_credentials", return_value=[]):
            run_eval_cp4_5.run_evaluation(
                self.manifest,
                self.scenarios_path,
                self.evaluations,
                self.log,
                spacing_seconds=0,
                evaluator_class=_FakeEvaluator,
            )
        changed = dict(self.generations[0])
        changed["generation_id"] = "other:0:1001"
        changed["prompt_id"] = "other"
        changed["sample_id"] = "other"
        self.manifest.write_text(
            json.dumps({
                "model": {
                    "name": "Qwen/Qwen3-1.7B",
                    "revision": run_eval_cp4_5.MODEL_REVISION,
                },
                "generations": [changed],
            }),
            encoding="utf-8",
        )
        with patch.object(run_eval_cp4_5, "discover_api_credentials", return_value=[]):
            with self.assertRaisesRegex(ValueError, "different generation cohort"):
                run_eval_cp4_5.run_evaluation(
                    self.manifest,
                    self.scenarios_path,
                    self.evaluations,
                    self.log,
                    spacing_seconds=0,
                    evaluator_class=_FakeEvaluator,
                )

    def test_run_log_allows_monotonic_generation_cohort_expansion(self):
        with patch.object(run_eval_cp4_5, "discover_api_credentials", return_value=[]):
            run_eval_cp4_5.run_evaluation(
                self.manifest,
                self.scenarios_path,
                self.evaluations,
                self.log,
                spacing_seconds=0,
                evaluator_class=_FakeEvaluator,
            )
        added = dict(self.generations[0])
        added.update({
            "generation_id": "sample:1:1001",
            "dataset_index": 1,
        })
        self.generations.append(added)
        self._write_inputs()
        self.scenarios_path.write_text(
            json.dumps([self.scenario, self.scenario]),
            encoding="utf-8",
        )
        with patch.object(run_eval_cp4_5, "discover_api_credentials", return_value=[]):
            result = run_eval_cp4_5.run_evaluation(
                self.manifest,
                self.scenarios_path,
                self.evaluations,
                self.log,
                spacing_seconds=0,
                evaluator_class=_FakeEvaluator,
            )
        self.assertEqual("completed", result["stop_reason"])
        self.assertEqual(2, result["completed_record_count"])
        self.assertEqual(
            ["sample:0:1001", "sample:1:1001"],
            _FakeEvaluator.calls,
        )

    def test_alternate_evaluator_profile_is_version_isolated_and_limited(self):
        result = run_eval_cp4_5.run_evaluation(
            generation_path=self.manifest,
            scenarios_path=self.scenarios_path,
            evaluations_dir=self.evaluations,
            log_path=self.log,
            spacing_seconds=0,
            evaluator_class=_FakeEvaluator,
            evaluator_model=run_eval_cp4_6.EVALUATOR_MODEL,
            prompt_version=run_eval_cp4_6.PROMPT_VERSION,
            change_reason=run_eval_cp4_6.CHANGE_REASON,
            message_builder=run_eval_cp4_6.build_messages,
            response_schema_builder=run_eval_cp4_6.response_schema,
            max_evaluations=1,
        )
        record_path = self.evaluations / run_eval_cp4_5._safe_generation_name(
            self.generations[0]["generation_id"]
        )
        record = json.loads(record_path.read_text(encoding="utf-8"))
        self.assertEqual("gemini-2.5-flash-lite", record["evaluator_model"])
        self.assertEqual("CP4.6.0", record["prompt_version"])
        self.assertEqual("evaluation_limit_reached", result["stop_reason"])
        self.assertEqual(1, result["completed_record_count"])

    def test_quota_recovery_appends_retry_without_replacing_failed_record(self):
        failed = EvaluationRecord(
            schema_version=SCHEMA_VERSION,
            evaluator_model=run_eval_cp4_5.EVALUATOR_MODEL,
            prompt_version=run_eval_cp4_5.PROMPT_VERSION,
            timestamp="2026-10-15T00:00:00+00:00",
            source_generation_id="sample:0:1001",
            dataset_id="squad_v2",
            evaluation_status=EvaluationStatus.FAILED,
            retry_count=0,
            generation_label=None,
            error_type="QuotaExhaustedError",
            attempts=[{
                "attempt": 1,
                "key_slot": "GEMINI_API_KEY",
                "status": "FAILED",
                "error_type": "QuotaExhaustedError",
            }],
        ).validate()
        original_path = self.evaluations / "sample_0_1001.json"
        original_path.parent.mkdir(parents=True)
        original_path.write_text(json.dumps(failed.to_dict()), encoding="utf-8")
        original_content = original_path.read_text(encoding="utf-8")
        self.log.write_text(json.dumps({
            "evaluator_model": run_eval_cp4_5.EVALUATOR_MODEL,
            "prompt_version": run_eval_cp4_5.PROMPT_VERSION,
            "started_at": "2026-10-15T00:00:00+00:00",
            "iterations": [],
            "unusable_credential_slots": {
                "GEMINI_API_KEY": "QuotaExhaustedError"
            },
            "quota_exhausted_at": {
                "GEMINI_API_KEY": "2026-10-15T00:00:00+00:00"
            },
        }), encoding="utf-8")

        with patch.object(
            run_eval_cp4_5, "discover_api_credentials",
            return_value=[("GEMINI_API_KEY", "test-only-key")],
        ):
            result = run_eval_cp4_5.run_evaluation(
                self.manifest,
                self.scenarios_path,
                self.evaluations,
                self.log,
                spacing_seconds=0,
                evaluator_class=_FakeEvaluator,
            )

        self.assertEqual(original_content, original_path.read_text(encoding="utf-8"))
        retry_files = list((self.evaluations / "retries").glob("*.json"))
        self.assertEqual(1, len(retry_files))
        retry = EvaluationRecord.from_dict(
            json.loads(retry_files[0].read_text(encoding="utf-8"))
        )
        self.assertEqual(EvaluationStatus.SUCCESS, retry.evaluation_status)
        self.assertEqual(
            "QuotaExhaustedError", retry.attempts[0]["error_type"]
        )
        self.assertEqual("completed", result["stop_reason"])
        self.assertEqual({"SUCCESS": 1}, result["status_counts"])

    def test_targets_require_ten_resolved_binary_labels(self):
        generation_rows = []
        self.generations = []
        outcomes = (
            [(EvaluationStatus.SUCCESS, GenerationLabel.RELIABLE)] * 10
            + [(EvaluationStatus.SUCCESS, GenerationLabel.RELIABLE)] * 7
            + [(EvaluationStatus.SUCCESS, GenerationLabel.INADEQUATE)]
            + [(EvaluationStatus.MANUAL_REVIEW, GenerationLabel.RELIABLE)]
            + [(EvaluationStatus.FAILED, None)]
        )
        scenarios = [self.scenario] * 2
        for index, (status, label) in enumerate(outcomes):
            prompt_index = index // 10
            generation = {
                "generation_id": f"sample{prompt_index}:{prompt_index}:{index}",
                "prompt_id": f"sample{prompt_index}",
                "dataset_index": prompt_index,
            }
            generation_rows.append(generation)
            record = EvaluationRecord(
                schema_version=SCHEMA_VERSION,
                evaluator_model=run_eval_cp4_5.EVALUATOR_MODEL,
                prompt_version=run_eval_cp4_5.PROMPT_VERSION,
                timestamp="2026-10-17T00:00:00+00:00",
                source_generation_id=generation["generation_id"],
                dataset_id="squad_v2",
                evaluation_status=status,
                retry_count=0,
                generation_label=label,
            ).validate()
            path = self.evaluations / run_eval_cp4_5._safe_generation_name(
                generation["generation_id"]
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(record.to_dict()), encoding="utf-8")

        self.manifest.write_text(
            json.dumps({
                "model": {
                    "name": "Qwen/Qwen3-1.7B",
                    "revision": run_eval_cp4_5.MODEL_REVISION,
                },
                "generations": generation_rows,
            }),
            encoding="utf-8",
        )
        targets = run_eval_cp4_5.build_targets(
            self.manifest, self.evaluations, self.root / "targets.json"
        )
        self.assertTrue(targets["targets"][0]["target_eligible"])
        self.assertEqual(0.0, targets["targets"][0]["empirical_hallucination_risk"])
        self.assertGreater(
            targets["targets"][0]["risk_wilson_95_interval"][1],
            0.0,
        )
        self.assertFalse(targets["targets"][1]["target_eligible"])
        self.assertIsNone(targets["targets"][1]["empirical_hallucination_risk"])
        self.assertEqual(1, targets["targets"][1]["n_inadequate"])
        self.assertEqual(1, targets["targets"][1]["n_manual_review"])
        self.assertEqual(1, targets["targets"][1]["n_failed"])

    def test_only_quota_blocks_expired_for_24_hours_are_reopened(self):
        now = datetime(2026, 10, 18, tzinfo=timezone.utc)
        log = {
            "unusable_credential_slots": {
                "GEMINI_API_KEY": "QuotaExhaustedError",
                "GEMINI_API_KEY_2": "CredentialAccessError",
                "GEMINI_API_KEY_3": "QuotaExhaustedError",
            },
            "quota_exhausted_at": {
                "GEMINI_API_KEY": "2026-10-16T23:00:00+00:00",
                "GEMINI_API_KEY_3": "2026-10-17T12:00:00+00:00",
            },
        }
        reopened = run_eval_cp4_5._reopen_expired_quota_slots(log, now)
        self.assertEqual(["GEMINI_API_KEY"], reopened)
        self.assertEqual(
            {
                "GEMINI_API_KEY_2": "CredentialAccessError",
                "GEMINI_API_KEY_3": "QuotaExhaustedError",
            },
            log["unusable_credential_slots"],
        )

    def test_short_rate_limit_cooldown_reopens_only_quota_slots(self):
        now = datetime(2026, 10, 18, 0, 1, tzinfo=timezone.utc)
        log = {
            "unusable_credential_slots": {
                "GEMINI_API_KEY": "QuotaExhaustedError",
                "GEMINI_API_KEY_2": "CredentialAccessError",
            },
            "quota_exhausted_at": {
                "GEMINI_API_KEY": "2026-10-18T00:00:00+00:00",
                "GEMINI_API_KEY_2": "2026-10-18T00:00:00+00:00",
            },
        }
        self.assertEqual(
            ["GEMINI_API_KEY"],
            run_eval_cp4_5._reopen_expired_quota_slots(
                log, now, cooldown=timedelta(seconds=60)
            ),
        )
        self.assertNotIn("GEMINI_API_KEY", log["unusable_credential_slots"])
        self.assertEqual(
            "CredentialAccessError",
            log["unusable_credential_slots"]["GEMINI_API_KEY_2"],
        )

    def test_restores_quota_cooldown_from_persisted_attempt_history(self):
        record = EvaluationRecord(
            schema_version=SCHEMA_VERSION,
            evaluator_model=run_eval_cp4_5.EVALUATOR_MODEL,
            prompt_version=run_eval_cp4_5.PROMPT_VERSION,
            timestamp="2026-10-17T12:00:00+00:00",
            source_generation_id="sample:0:1001",
            dataset_id="squad_v2",
            evaluation_status=EvaluationStatus.SUCCESS,
            retry_count=1,
            generation_label=GenerationLabel.RELIABLE,
            attempts=[
                {
                    "key_slot": "GEMINI_API_KEY",
                    "error_type": "QuotaExhaustedError",
                    "status": "FAILED",
                }
            ],
        ).validate()
        path = self.evaluations / "sample_0_1001.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(record.to_dict()), encoding="utf-8")
        later = EvaluationRecord.from_dict(record.to_dict())
        later.timestamp = "2026-10-17T13:00:00+00:00"
        later.source_generation_id = "sample:0:1002"
        (self.evaluations / "sample_0_1002.json").write_text(
            json.dumps(later.to_dict()), encoding="utf-8"
        )
        log = {
            "unusable_credential_slots": {
                "GEMINI_API_KEY": "QuotaExhaustedError"
            }
        }
        run_eval_cp4_5._restore_quota_timestamps(log, self.evaluations)
        self.assertEqual(
            "2026-10-17T13:00:00+00:00",
            log["quota_exhausted_at"]["GEMINI_API_KEY"],
        )


class TestPredictorUncertainty(unittest.TestCase):
    def test_bootstrap_intervals_are_deterministic_and_zero_for_perfect_fit(self):
        first = _bootstrap_metric_intervals(
            [0.0, 0.5, 1.0],
            [0.0, 0.5, 1.0],
            [0.0, 0.5, 1.0],
        )
        second = _bootstrap_metric_intervals(
            [0.0, 0.5, 1.0],
            [0.0, 0.5, 1.0],
            [0.0, 0.5, 1.0],
        )
        self.assertEqual(first, second)
        self.assertEqual([0.0, 0.0], first["mae_95_percentile_interval"])
        self.assertEqual([0.0, 0.0], first["rmse_95_percentile_interval"])
        self.assertEqual(
            [0.0, 0.0],
            first["mae_improvement_over_baseline_95_percentile_interval"],
        )


if __name__ == "__main__":
    unittest.main()
