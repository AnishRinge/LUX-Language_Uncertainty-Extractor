"""Focused tests for the LUX production dataset generation pipeline.

Covers:
  * Deterministic / stratified prompt selection
  * Production generation-record building (no labels attached pre-eval)
  * Crash-safe atomic persistence
  * Predictor-feature separation (no generated answers / labels / risk)
  * Empirical-risk computation (frozen CP4.4 formula)
  * Resume / checkpoint behaviour
  * Evaluation-request building (only Q/C/A sent to evaluator)
"""

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import run_production as prod  # noqa: E402
import run_cp4_2_pilot as cp42  # noqa: E402
from src.evaluation.schemas import (  # noqa: E402
    SCHEMA_VERSION,
    EvaluationRecord,
    EvaluationStatus,
    GenerationLabel,
)
from src.evaluation.prompt import PROMPT_VERSION  # noqa: E402
from src.evaluation.evaluator_runner import build_request_from_cp4_2_5  # noqa: E402

STATUS_SUCCESS = cp42.STATUS_SUCCESS
STATUS_FAILED = cp42.STATUS_FAILED


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_scenario(idx, answerable=True):
    return {
        "sample_id": "squad_v2_{0}".format(idx),
        "source": {
            "dataset": "squad_v2",
            "split": "train",
            "original_id": "orig_{0}".format(idx),
            "article_id": "0",
            "paragraph_id": "0",
            "article_title": "Test Article",
        },
        "prompt": {
            "question": "What is question {0}?".format(idx),
            "context": "The context for question {0} states the answer is 42.".format(idx),
        },
        "reference": {
            "answerability": "ANSWERABLE" if answerable else "UNANSWERABLE_FROM_CONTEXT",
            "answers": [{"text": ["42"], "answer_id": 0}] if answerable else [],
            "supporting_information": None,
        },
        "metadata": {
            "phenomenon": "grounded_contextual_reliability",
            "expected_behavior": "ANSWER" if answerable else "ABSTAIN",
            "source_metadata": {"is_impossible": not answerable},
        },
    }


def _make_scenarios(n=20, split_at=10):
    return [_make_scenario(i, answerable=(i < split_at)) for i in range(n)]


def _make_eval_record(gen_id, status, label=None, retry_count=0):
    """Build a valid EvaluationRecord for testing."""
    if status == EvaluationStatus.FAILED:
        label = None
    elif status == EvaluationStatus.MANUAL_REVIEW:
        label = GenerationLabel.INADEQUATE
    elif status == EvaluationStatus.SUCCESS and label is None:
        label = GenerationLabel.RELIABLE

    return EvaluationRecord(
        schema_version=SCHEMA_VERSION,
        evaluator_model="test-evaluator",
        prompt_version=PROMPT_VERSION,
        timestamp="2026-01-01T00:00:00Z",
        source_generation_id=gen_id,
        dataset_id="test",
        evaluation_status=status,
        retry_count=retry_count,
        claims=[],
        generation_label=label,
    ).validate()


def _write_eval_record(directory, gen_id, status, label=None):
    """Write an evaluation record to *directory*."""
    record = _make_eval_record(gen_id, status, label)
    safe = gen_id.replace(":", "_").replace("/", "_")
    path = directory / "{0}.json".format(safe)
    prod.atomic_write_json(path, record.to_dict())
    return path


# ---------------------------------------------------------------------------
# Selection tests
# ---------------------------------------------------------------------------
class TestSelectProductionScenarios(unittest.TestCase):
    def setUp(self):
        self.scenarios = _make_scenarios(n=20, split_at=10)

    def test_select_all_when_num_prompts_is_none(self):
        selected = prod.select_production_scenarios(self.scenarios, None)
        self.assertEqual(len(selected), 20)

    def test_select_fewer_when_num_prompts_specified(self):
        selected = prod.select_production_scenarios(self.scenarios, 10)
        self.assertEqual(len(selected), 10)

    def test_selection_preserves_answerable_ratio(self):
        selected = prod.select_production_scenarios(self.scenarios, 10)
        n_ans = sum(1 for _, s in selected if cp42.is_answerable(s))
        n_unans = 10 - n_ans
        # 10 of 20 (5 answerable, 5 unanswerable) → 5/5 split.
        self.assertEqual(n_ans, 5)
        self.assertEqual(n_unans, 5)

    def test_selection_is_deterministic(self):
        run1 = prod.select_production_scenarios(self.scenarios, 12)
        run2 = prod.select_production_scenarios(self.scenarios, 12)
        self.assertEqual(
            [idx for idx, _ in run1],
            [idx for idx, _ in run2],
        )

    def test_select_more_than_available_returns_all(self):
        selected = prod.select_production_scenarios(self.scenarios, 100)
        self.assertEqual(len(selected), 20)

    def test_select_one_from_each_class(self):
        scenarios = [_make_scenario(0, answerable=True),
                     _make_scenario(1, answerable=True),
                     _make_scenario(2, answerable=False)]
        selected = prod.select_production_scenarios(scenarios, 2)
        self.assertEqual(len(selected), 2)
        n_ans = sum(1 for _, s in selected if cp42.is_answerable(s))
        n_unans = 2 - n_ans
        self.assertEqual(n_ans, 1)
        self.assertEqual(n_unans, 1)


# ---------------------------------------------------------------------------
# Record building tests
# ---------------------------------------------------------------------------
class TestBuildProductionRecord(unittest.TestCase):
    def setUp(self):
        self.scenario = _make_scenario(42, answerable=True)

    def test_record_has_all_required_fields(self):
        rec = prod.build_production_record(
            scenario=self.scenario,
            dataset_index=42,
            generation_index=0,
            seed=1001,
            prompt_text="rendered prompt",
            prompt_token_count=20,
            status=STATUS_SUCCESS,
            generated_answer="42",
            generated_token_count=3,
            hidden_state_ref="data/production/hidden_states/hs_42.json",
        )
        required = [
            "prompt_id", "dataset_source", "dataset_index", "question", "context",
            "answerability", "generation_id", "seed", "generated_answer",
            "generation_status", "hidden_state_reference", "evaluation_status",
            "generation_label", "claim_evaluations", "empirical_hallucination_risk",
            "provenance", "model_name", "model_revision",
        ]
        for field in required:
            self.assertIn(field, rec, msg="Missing required field: {0}".format(field))

    def test_record_has_no_evaluation_labels_initially(self):
        rec = prod.build_production_record(
            scenario=self.scenario,
            dataset_index=42,
            generation_index=0,
            seed=1001,
            prompt_text="test",
            prompt_token_count=10,
            status=STATUS_SUCCESS,
            generated_answer="42",
            hidden_state_ref="hs_42.json",
        )
        self.assertIsNone(rec["evaluation_status"])
        self.assertIsNone(rec["generation_label"])
        self.assertIsNone(rec["claim_evaluations"])
        self.assertIsNone(rec["empirical_hallucination_risk"])

    def test_record_preserves_frozen_protocol(self):
        rec = prod.build_production_record(
            scenario=self.scenario,
            dataset_index=42,
            generation_index=0,
            seed=1001,
            prompt_text="test",
            prompt_token_count=10,
            status=STATUS_SUCCESS,
            generated_answer="42",
            hidden_state_ref="hs_42.json",
        )
        self.assertEqual(rec["model_name"], cp42.MODEL_NAME)
        self.assertEqual(rec["model_revision"], cp42.MODEL_REVISION)
        self.assertEqual(rec["device"], cp42.DEVICE)
        self.assertEqual(rec["thinking_mode"], cp42.THINKING_MODE)
        self.assertEqual(rec["do_sample"], cp42.DO_SAMPLE)
        self.assertEqual(rec["temperature"], cp42.TEMPERATURE)
        self.assertEqual(rec["top_p"], cp42.TOP_P)
        self.assertEqual(rec["top_k"], cp42.TOP_K)

    def test_generation_id_is_stable(self):
        rec = prod.build_production_record(
            scenario=self.scenario,
            dataset_index=42,
            generation_index=0,
            seed=1001,
            prompt_text="test",
            prompt_token_count=10,
            status=STATUS_SUCCESS,
            generated_answer="42",
            hidden_state_ref="hs_42.json",
        )
        self.assertEqual(rec["generation_id"], "squad_v2_42:42:1001")

    def test_failed_generation_is_marked(self):
        rec = prod.build_production_record(
            scenario=self.scenario,
            dataset_index=42,
            generation_index=0,
            seed=1001,
            prompt_text="test",
            prompt_token_count=10,
            status=STATUS_FAILED,
            error="OOM on CPU",
            hidden_state_ref="hs_42.json",
        )
        self.assertEqual(rec["generation_status"], STATUS_FAILED)
        self.assertEqual(rec["error"], "OOM on CPU")
        self.assertEqual(rec["generated_answer"], "")

    def test_record_compatible_with_eval_request_builder(self):
        rec = prod.build_production_record(
            scenario=self.scenario,
            dataset_index=42,
            generation_index=0,
            seed=1001,
            prompt_text="test",
            prompt_token_count=10,
            status=STATUS_SUCCESS,
            generated_answer="42",
            hidden_state_ref="hs_42.json",
        )
        req = build_request_from_cp4_2_5(rec, self.scenario)
        self.assertEqual(req.question, self.scenario["prompt"]["question"])
        self.assertEqual(req.context, self.scenario["prompt"]["context"])
        self.assertEqual(req.generated_answer, "42")


# ---------------------------------------------------------------------------
# Atomic write tests
# ---------------------------------------------------------------------------
class TestAtomicWrite(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmpdir, ignore_errors=True)
        self.tmp = Path(self.tmpdir)

    def test_atomic_write_creates_valid_json(self):
        path = self.tmp / "output" / "data.json"
        prod.atomic_write_json(path, {"key": "value", "n": 42})
        self.assertTrue(path.exists())
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        self.assertEqual(data, {"key": "value", "n": 42})

    def test_atomic_write_creates_parent_dirs(self):
        path = self.tmp / "a" / "b" / "c" / "deep.json"
        prod.atomic_write_json(path, {"x": 1})
        self.assertTrue(path.exists())

    def test_atomic_write_no_tmp_file_left(self):
        path = self.tmp / "data.json"
        prod.atomic_write_json(path, {"x": 1})
        tmp_files = list(self.tmp.glob("*.tmp"))
        self.assertEqual(len(tmp_files), 0)


# ---------------------------------------------------------------------------
# Predictor-feature separation tests (critical guarantee)
# ---------------------------------------------------------------------------
class TestFeatureSeparation(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmpdir, ignore_errors=True)
        self.tmp = Path(self.tmpdir)
        self._patches = []
        for attr in ["PREDICTOR_FEATURES_PATH"]:
            orig = getattr(prod, attr)
            self._patches.append(
                patch.object(prod, attr, self.tmp / Path(orig).name)
            )
        for p in self._patches:
            p.start()
        self.addCleanup(lambda: [p.stop() for p in self._patches])

    def _make_generation_with_full_eval(self):
        """Build a generation record simulating post-evaluation state."""
        scenario = _make_scenario(7, answerable=True)
        gen = prod.build_production_record(
            scenario=scenario,
            dataset_index=7,
            generation_index=0,
            seed=1001,
            prompt_text="test prompt",
            prompt_token_count=15,
            status=STATUS_SUCCESS,
            generated_answer="The answer is 42 because it is.",
            generated_token_count=7,
            latency=2.5,
            hidden_state_ref="data/production/hidden_states/hs_7.json",
        )
        # Simulate post-evaluation state — these must NOT leak into features.
        gen["evaluation_status"] = "SUCCESS"
        gen["generation_label"] = "UNRELIABLE"
        gen["claim_evaluations"] = [
            {"claim_text": "42", "claim_status": "CONTRADICTED"},
        ]
        gen["empirical_hallucination_risk"] = 0.67
        gen["dataset_source"] = "squad_v2"
        gen["status"] = "SUCCESS"
        gen["latency"] = 2.5
        return gen

    def _build(self):
        gen = self._make_generation_with_full_eval()
        return prod.build_predictor_features([gen], tokenizer=None)

    def test_exclude_generated_answers(self):
        blob = json.dumps(self._build()["predictor_features"][0])
        self.assertNotIn("generated_answer", blob)
        self.assertNotIn("The answer is 42", blob)

    def test_exclude_generation_labels(self):
        blob = json.dumps(self._build()["predictor_features"][0])
        self.assertNotIn("generation_label", blob)
        self.assertNotIn("UNRELIABLE", blob)

    def test_exclude_risk_values(self):
        blob = json.dumps(self._build()["predictor_features"][0])
        self.assertNotIn("empirical_hallucination_risk", blob)
        self.assertNotIn("0.67", blob)

    def test_exclude_dataset_source(self):
        blob = json.dumps(self._build()["predictor_features"][0])
        self.assertNotIn("dataset_source", blob)

    def test_exclude_eval_status(self):
        blob = json.dumps(self._build()["predictor_features"][0])
        self.assertNotIn("evaluation_status", blob)

    def test_exclude_claim_evaluations(self):
        blob = json.dumps(self._build()["predictor_features"][0])
        self.assertNotIn("claim_evaluations", blob)

    def test_exclude_latency_and_status(self):
        blob = json.dumps(self._build()["predictor_features"][0])
        self.assertNotIn("latency", blob)
        self.assertNotIn('"status"', blob)

    def test_include_hidden_state_ref(self):
        entry = self._build()["predictor_features"][0]
        self.assertEqual(
            entry["hidden_state_ref"],
            "data/production/hidden_states/hs_7.json",
        )

    def test_include_prompt_metrics(self):
        entry = self._build()["predictor_features"][0]
        self.assertIn("prompt_metrics", entry)
        self.assertIn("question_chars", entry["prompt_metrics"])
        self.assertIn("context_chars", entry["prompt_metrics"])
        self.assertTrue(entry["prompt_metrics"]["question_chars"] > 0)

    def test_dedup_by_dataset_index(self):
        gen1 = self._make_generation_with_full_eval()
        gen2 = self._make_generation_with_full_eval()
        gen2["seed"] = 1002
        gen2["generation_id"] = "squad_v2_7:7:1002"
        features = prod.build_predictor_features([gen1, gen2], tokenizer=None)
        self.assertEqual(len(features["predictor_features"]), 1)

    def test_excluded_fields_documented(self):
        features = self._build()
        excluded = features["pipeline"]["excluded_fields"]
        for field in ["generated_answer", "generation_label",
                       "empirical_hallucination_risk", "dataset_source"]:
            self.assertIn(field, excluded)


# ---------------------------------------------------------------------------
# Risk computation tests
# ---------------------------------------------------------------------------
class TestRiskComputation(unittest.TestCase):
    def test_risk_ratio_is_correct(self):
        records = [
            _make_eval_record("g1", EvaluationStatus.SUCCESS, GenerationLabel.UNRELIABLE),
            _make_eval_record("g2", EvaluationStatus.SUCCESS, GenerationLabel.UNRELIABLE),
            _make_eval_record("g3", EvaluationStatus.SUCCESS, GenerationLabel.UNRELIABLE),
            _make_eval_record("g4", EvaluationStatus.SUCCESS, GenerationLabel.RELIABLE),
            _make_eval_record("g5", EvaluationStatus.SUCCESS, GenerationLabel.RELIABLE),
            _make_eval_record("g6", EvaluationStatus.FAILED),
            _make_eval_record("g7", EvaluationStatus.MANUAL_REVIEW),
        ]
        result = prod.compute_per_prompt_risk(records)
        self.assertAlmostEqual(result["empirical_hallucination_risk"], 0.6, places=6)
        self.assertEqual(result["n_resolved"], 5)
        self.assertEqual(result["n_unreliable"], 3)

    def test_failed_excluded_from_denominator(self):
        records = [
            _make_eval_record("g1", EvaluationStatus.SUCCESS, GenerationLabel.UNRELIABLE),
            _make_eval_record("g2", EvaluationStatus.FAILED),
        ]
        result = prod.compute_per_prompt_risk(records)
        self.assertEqual(result["empirical_hallucination_risk"], 1.0)
        self.assertEqual(result["n_resolved"], 1)

    def test_manual_review_excluded_from_denominator(self):
        records = [
            _make_eval_record("g1", EvaluationStatus.SUCCESS, GenerationLabel.RELIABLE),
            _make_eval_record("g2", EvaluationStatus.MANUAL_REVIEW),
        ]
        result = prod.compute_per_prompt_risk(records)
        self.assertEqual(result["empirical_hallucination_risk"], 0.0)
        self.assertEqual(result["n_resolved"], 1)

    def test_risk_is_none_when_no_successful(self):
        records = [
            _make_eval_record("g1", EvaluationStatus.FAILED),
            _make_eval_record("g2", EvaluationStatus.MANUAL_REVIEW),
        ]
        result = prod.compute_per_prompt_risk(records)
        self.assertIsNone(result["empirical_hallucination_risk"])
        self.assertEqual(result["n_resolved"], 0)

    def test_failed_record_has_no_label(self):
        """FAILED evaluations must never carry a generation label."""
        record = _make_eval_record("g1", EvaluationStatus.FAILED)
        self.assertIsNone(record.generation_label)
        with self.assertRaises(ValueError):
            EvaluationRecord(
                schema_version=SCHEMA_VERSION,
                evaluator_model="test",
                prompt_version=PROMPT_VERSION,
                timestamp="2026-01-01T00:00:00Z",
                source_generation_id="g1",
                dataset_id="test",
                evaluation_status=EvaluationStatus.FAILED,
                retry_count=0,
                claims=[],
                generation_label=GenerationLabel.UNRELIABLE,
            ).validate()


# ---------------------------------------------------------------------------
# Build targets tests
# ---------------------------------------------------------------------------
class TestBuildTargets(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmpdir, ignore_errors=True)
        self.tmp = Path(self.tmpdir)
        self.eval_dir = self.tmp / "evaluations"
        self.eval_dir.mkdir(parents=True, exist_ok=True)
        self._patches = []
        for attr in ["EVALUATIONS_DIR", "TARGETS_PATH", "GENERATIONS_PATH",
                     "PROD_DIR"]:
            orig = getattr(prod, attr)
            p = patch.object(prod, attr, self.tmp / Path(orig).name)
            p.start()
            self._patches.append(p)
        self.addCleanup(lambda: [p.stop() for p in self._patches])

    def _make_generation(self, dataset_index, seed, eval_status=None,
                         gen_label=None):
        scenario = _make_scenario(dataset_index, answerable=True)
        gen = prod.build_production_record(
            scenario=scenario,
            dataset_index=dataset_index,
            generation_index=0,
            seed=seed,
            prompt_text="test prompt",
            prompt_token_count=10,
            status=STATUS_SUCCESS,
            generated_answer="42",
            hidden_state_ref="hs_{0}.json".format(dataset_index),
        )
        if eval_status is not None:
            gen["evaluation_status"] = eval_status
            gen["generation_label"] = gen_label
        gen["generation_id"] = "squad_v2_{0}:{1}:{2}".format(
            dataset_index, dataset_index, seed
        )
        return gen

    def test_targets_computed_from_evaluation_records(self):
        gen1 = self._make_generation(0, 1001, "SUCCESS", "UNRELIABLE")
        gen2 = self._make_generation(0, 1002, "SUCCESS", "RELIABLE")
        generations = [gen1, gen2]

        _write_eval_record(self.eval_dir, gen1["generation_id"],
                           EvaluationStatus.SUCCESS, GenerationLabel.UNRELIABLE)
        _write_eval_record(self.eval_dir, gen2["generation_id"],
                           EvaluationStatus.SUCCESS, GenerationLabel.RELIABLE)

        targets = prod.build_targets(generations)
        self.assertEqual(len(targets), 1)
        self.assertAlmostEqual(targets[0]["empirical_hallucination_risk"], 0.5, places=6)
        self.assertEqual(targets[0]["n_resolved"], 2)
        self.assertEqual(targets[0]["n_unreliable"], 1)
        self.assertEqual(gen1["empirical_hallucination_risk"], 0.5)
        self.assertEqual(gen2["empirical_hallucination_risk"], 0.5)

    def test_targets_none_when_no_evaluations(self):
        gen1 = self._make_generation(0, 1001)
        gen2 = self._make_generation(0, 1002)
        generations = [gen1, gen2]
        targets = prod.build_targets(generations)
        self.assertEqual(len(targets), 1)
        self.assertIsNone(targets[0]["empirical_hallucination_risk"])

    def test_failed_evaluations_excluded_from_risk(self):
        gen1 = self._make_generation(0, 1001)
        gen2 = self._make_generation(0, 1002)
        generations = [gen1, gen2]

        _write_eval_record(self.eval_dir, gen1["generation_id"],
                           EvaluationStatus.FAILED)
        _write_eval_record(self.eval_dir, gen2["generation_id"],
                           EvaluationStatus.SUCCESS, GenerationLabel.UNRELIABLE)

        targets = prod.build_targets(generations)
        self.assertAlmostEqual(targets[0]["empirical_hallucination_risk"], 1.0, places=6)

    def test_targets_multiple_prompts(self):
        gen1 = self._make_generation(0, 1001)
        gen2 = self._make_generation(1, 1001)
        gen3 = self._make_generation(1, 1002)
        generations = [gen1, gen2, gen3]

        _write_eval_record(self.eval_dir, gen1["generation_id"],
                           EvaluationStatus.SUCCESS, GenerationLabel.RELIABLE)
        _write_eval_record(self.eval_dir, gen2["generation_id"],
                           EvaluationStatus.SUCCESS, GenerationLabel.UNRELIABLE)
        _write_eval_record(self.eval_dir, gen3["generation_id"],
                           EvaluationStatus.SUCCESS, GenerationLabel.UNRELIABLE)

        targets = prod.build_targets(generations)
        self.assertEqual(len(targets), 2)
        by_idx = {t["dataset_index"]: t for t in targets}
        self.assertAlmostEqual(by_idx[0]["empirical_hallucination_risk"], 0.0)
        self.assertAlmostEqual(by_idx[1]["empirical_hallucination_risk"], 1.0)


# ---------------------------------------------------------------------------
# Evaluation request building tests
# ---------------------------------------------------------------------------
class TestEvalRequestBuilder(unittest.TestCase):
    def test_only_question_context_answer_sent(self):
        from src.evaluation.prompt import build_messages

        scenario = _make_scenario(3, answerable=True)
        gen = prod.build_production_record(
            scenario=scenario,
            dataset_index=3,
            generation_index=0,
            seed=1001,
            prompt_text="test",
            prompt_token_count=10,
            status=STATUS_SUCCESS,
            generated_answer="The answer is 42.",
            hidden_state_ref="hs_3.json",
        )
        req = build_request_from_cp4_2_5(gen, scenario)
        messages = build_messages(req.question, req.context, req.generated_answer)

        user_msg = next(m for m in messages if m["role"] == "user")
        self.assertIn(req.question, user_msg["content"])
        self.assertIn(req.context, user_msg["content"])
        self.assertIn(req.generated_answer, user_msg["content"])

        blob = json.dumps(messages)
        self.assertNotIn(str(req.seed), blob)
        self.assertNotIn(req.model_name, blob)
        self.assertNotIn(req.model_revision, blob)

    def test_skips_already_evaluated(self):
        scenario = _make_scenario(5)
        gen_evaluated = prod.build_production_record(
            scenario=scenario, dataset_index=5, generation_index=0,
            seed=1001, prompt_text="t", prompt_token_count=5,
            status=STATUS_SUCCESS, generated_answer="42",
            hidden_state_ref="hs_5.json",
        )
        gen_evaluated["evaluation_status"] = "SUCCESS"
        gen_evaluated["generation_label"] = "RELIABLE"
        gen_evaluated["generation_id"] = "squad_v2_5:5:1001"

        gen_pending = prod.build_production_record(
            scenario=scenario, dataset_index=5, generation_index=1,
            seed=1002, prompt_text="t", prompt_token_count=5,
            status=STATUS_SUCCESS, generated_answer="43",
            hidden_state_ref="hs_5.json",
        )
        gen_pending["generation_id"] = "squad_v2_5:5:1002"

        scenarios_map = {5: scenario}
        pending = prod._eval_requests_from_generations(
            [gen_evaluated, gen_pending], scenarios_map
        )
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0][0]["generation_id"], "squad_v2_5:5:1002")


# ---------------------------------------------------------------------------
# Path format tests
# ---------------------------------------------------------------------------
class TestPathFormats(unittest.TestCase):
    def test_hidden_state_path(self):
        p = prod._hidden_state_path(19919)
        self.assertEqual(p.name, "hs_19919.json")
        self.assertIn("hidden_states", str(p))

    def test_eval_record_path(self):
        gen_id = "squad_v2_sample19919:19919:1001"
        p = prod._generation_evaluation_path(gen_id)
        self.assertEqual(p.name, "squad_v2_sample19919_19919_1001.json")
        self.assertIn("evaluations", str(p))


# ---------------------------------------------------------------------------
# Checkpoint / resume tests
# ---------------------------------------------------------------------------
class TestCheckpointResume(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmpdir, ignore_errors=True)
        self._patches = []
        for attr in ["CHECKPOINT_PATH", "PROD_DIR", "GENERATIONS_PATH",
                     "HIDDEN_STATES_DIR", "EVALUATIONS_DIR", "TARGETS_PATH",
                     "PREDICTOR_FEATURES_PATH"]:
            orig = getattr(prod, attr)
            p = patch.object(prod, attr, Path(self.tmpdir) / Path(orig).name)
            p.start()
            self._patches.append(p)
        self.addCleanup(lambda: [p.stop() for p in self._patches])

    def test_load_checkpoint_initialises_empty(self):
        ckpt = prod.load_checkpoint()
        self.assertEqual(ckpt["phase"], "not_started")
        self.assertEqual(ckpt["completed_prompts"], [])
        self.assertEqual(ckpt["all_generations"], [])

    def test_save_then_load_checkpoint_roundtrips(self):
        ckpt = prod.load_checkpoint()
        ckpt["phase"] = "generation"
        ckpt["completed_prompts"] = [42, 67]
        ckpt["all_generations"] = [{"dataset_index": 42, "seed": 1001}]
        prod.save_checkpoint(ckpt)

        loaded = prod.load_checkpoint()
        self.assertEqual(loaded["phase"], "generation")
        self.assertEqual(loaded["completed_prompts"], [42, 67])
        self.assertEqual(len(loaded["all_generations"]), 1)

    def test_checkpoint_tracks_completed_prompts(self):
        ckpt = prod.load_checkpoint()
        ckpt["completed_prompts"] = [10, 20]
        ckpt["all_generations"] = [
            {"dataset_index": 10, "seed": 1001, "generation_id": "g10:10:1001"},
            {"dataset_index": 10, "seed": 1002, "generation_id": "g10:10:1002"},
            {"dataset_index": 20, "seed": 1001, "generation_id": "g20:20:1001"},
        ]
        prod.save_checkpoint(ckpt)
        loaded = prod.load_checkpoint()
        self.assertIn(10, loaded["completed_prompts"])
        self.assertIn(20, loaded["completed_prompts"])
        self.assertEqual(len(loaded["all_generations"]), 3)


# ---------------------------------------------------------------------------
# Module structure tests
# ---------------------------------------------------------------------------
class TestModuleStructure(unittest.TestCase):
    def test_module_imports(self):
        self.assertTrue(hasattr(prod, "run_generation_phase"))
        self.assertTrue(hasattr(prod, "run_evaluation_phase"))
        self.assertTrue(hasattr(prod, "build_targets_and_features"))
        self.assertTrue(hasattr(prod, "build_predictor_features"))
        self.assertTrue(hasattr(prod, "compute_per_prompt_risk"))
        self.assertTrue(hasattr(prod, "select_production_scenarios"))

    def test_model_constants_frozen(self):
        self.assertEqual(prod.MODEL_NAME, "Qwen/Qwen3-1.7B")
        self.assertEqual(
            prod.MODEL_REVISION,
            "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
        )
        self.assertEqual(prod.DEVICE, "cpu")
        self.assertEqual(prod.THINKING_MODE, False)
        self.assertEqual(prod.DEFAULT_GENERATIONS_PER_PROMPT, len(cp42.SEEDS))


if __name__ == "__main__":
    unittest.main()
