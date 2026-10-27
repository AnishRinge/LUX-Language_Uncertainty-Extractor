import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import run_eval_autonomous


class TestShouldStopForError(unittest.TestCase):
    def test_successful_fallback_with_prior_quota_does_not_stop(self):
        record = {
            "evaluation_status": "SUCCESS",
            "error_type": "",
            "attempts": [
                {"error_type": "QuotaExhaustedError", "status": "FAILED"},
                {"status": "SUCCESS", "key_slot": "GEMINI_API_KEY_2"},
            ],
        }
        self.assertEqual((False, ""), run_eval_autonomous.should_stop_for_error(record))

    def test_terminal_quota_error_stops(self):
        record = {
            "evaluation_status": "FAILED",
            "error_type": "QuotaExhaustedError",
            "attempts": [{"error_type": "QuotaExhaustedError", "status": "FAILED"}],
        }
        self.assertEqual(
            (True, "quota_exhausted"),
            run_eval_autonomous.should_stop_for_error(record),
        )

    def test_terminal_model_access_error_is_not_misreported_as_quota(self):
        record = {
            "evaluation_status": "FAILED",
            "error_type": "PersistentEvaluationError",
            "attempts": [
                {"error_type": "QuotaExhaustedError", "status": "FAILED"},
                {
                    "error_type": "PersistentEvaluationError",
                    "error_message": "404 NOT_FOUND: model unavailable",
                    "status": "FAILED",
                },
            ],
        }
        self.assertEqual(
            (True, "persistent_error"),
            run_eval_autonomous.should_stop_for_error(record),
        )

    def test_unusable_slot_state_tracks_specific_errors_and_later_success(self):
        record = {
            "attempts": [
                {
                    "key_slot": "GEMINI_API_KEY",
                    "status": "FAILED",
                    "error_type": "QuotaExhaustedError",
                },
                {
                    "key_slot": "GEMINI_API_KEY_2",
                    "status": "FAILED",
                    "error_type": "CredentialAccessError",
                },
                {
                    "key_slot": "GEMINI_API_KEY_3",
                    "status": "FAILED",
                    "error_type": "GlobalModelConfigurationError",
                },
            ]
        }
        known = run_eval_autonomous._update_unusable_slots(record, {})
        self.assertEqual(
            {
                "GEMINI_API_KEY": "QuotaExhaustedError",
                "GEMINI_API_KEY_2": "CredentialAccessError",
            },
            known,
        )
        recovered = run_eval_autonomous._update_unusable_slots(
            {
                "attempts": [
                    {"key_slot": "GEMINI_API_KEY_2", "status": "SUCCESS"}
                ]
            },
            known,
        )
        self.assertEqual({"GEMINI_API_KEY": "QuotaExhaustedError"}, recovered)

    def test_subprocess_receives_only_known_unusable_slot_names(self):
        completed = SimpleNamespace(returncode=0, stdout="", stderr="")
        with patch.object(
            run_eval_autonomous.subprocess, "run", return_value=completed
        ) as run:
            run_eval_autonomous.run_one_evaluation(
                {"GEMINI_API_KEY", "GEMINI_API_KEY_2"}
            )
        child_env = run.call_args.kwargs["env"]
        self.assertEqual(
            "GEMINI_API_KEY,GEMINI_API_KEY_2",
            child_env["GEMINI_SKIP_CREDENTIAL_SLOTS"],
        )

    def test_attempt_delta_excludes_prior_failed_runs_for_same_generation(self):
        log = {
            "iterations": [
                {"gen_id": "sample:1:1001", "attempts": 1},
                {"gen_id": "sample:1:1001", "attempts": 3},
                {"gen_id": "other:2:1001", "attempts": 1},
            ]
        }
        attempts = [
            {"attempt": index, "key_slot": f"GEMINI_API_KEY_{index}"}
            for index in range(1, 11)
        ]
        current = run_eval_autonomous._attempts_this_run(
            log, "sample:1:1001", attempts
        )
        self.assertEqual(7, len(current))
        self.assertEqual("GEMINI_API_KEY_4", current[0]["key_slot"])

    def test_report_derives_attempt_delta_for_legacy_run_entries(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            report_path = Path(temp_dir) / "report.md"
            eval_dir = Path(temp_dir) / "evaluations"
            eval_dir.mkdir()
            log = {
                "started_at": "2026-10-17T10:00:00+00:00",
                "session_budget_seconds": 9000,
                "elapsed_seconds": 15.7,
                "stop_reason": "persistent_error",
                "iterations": [
                    {
                        "timestamp": "2026-10-17T09:00:00+00:00",
                        "gen_id": "sample:1:1001",
                        "status": "FAILED",
                        "attempts": 3,
                        "credential_slots": [
                            "GEMINI_API_KEY",
                            "GEMINI_API_KEY_2",
                            "GEMINI_API_KEY_3",
                        ],
                        "exit_code": 0,
                    },
                    {
                        "timestamp": "2026-10-17T10:00:01+00:00",
                        "gen_id": "sample:1:1001",
                        "status": "FAILED",
                        "attempts": 10,
                        "credential_slots": [
                            "GEMINI_API_KEY",
                            "GEMINI_API_KEY_2",
                            "GEMINI_API_KEY_3",
                            "GEMINI_API_KEY_4",
                            "GEMINI_API_KEY_5",
                            "GEMINI_API_KEY_6",
                            "GEMINI_API_KEY_7",
                            "GEMINI_API_KEY_8",
                            "GEMINI_API_KEY_9",
                            "GEMINI_API_KEY_10",
                        ],
                        "exit_code": 0,
                    },
                ],
            }
            with (
                patch.object(run_eval_autonomous, "REPORT_PATH", report_path),
                patch.object(run_eval_autonomous, "EVAL_DIR", eval_dir),
            ):
                run_eval_autonomous._write_report(
                    log,
                    {
                        "SUCCESS": 34,
                        "FAILED": 1,
                        "MANUAL_REVIEW": 0,
                        "UNKNOWN": 0,
                    },
                    1,
                    "persistent_error",
                    276,
                )

            report = report_path.read_text(encoding="utf-8")
            self.assertIn("attempts this run=7", report)
            self.assertIn("total recorded attempts=10", report)
            self.assertIn("credential slots=GEMINI_API_KEY_10", report)
            self.assertIn("exit code=0", report)


if __name__ == "__main__":
    unittest.main()
