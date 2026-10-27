import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_expanded_cohort as expanded
import run_eval_cp4_7 as eval_cp47
import run_eval_cp4_5 as eval_runner
from src.evaluation.schemas import (
    EvaluationRecord,
    EvaluationStatus,
    GenerationLabel,
    SCHEMA_VERSION,
)
from src.production.expanded_cohort import (
    EVALUATOR_MODEL,
    BATCH_SIZE,
    MODEL_NAME,
    MODEL_REVISION,
    PROMPT_VERSION,
    SEEDS,
    build_preflight_report,
    batch_scenarios,
    create_selection_artifact,
    create_predictor_feature_artifact,
    load_json,
    select_expanded_scenarios,
    selection_digest,
)


def _make_scenario(index):
    return {
        "sample_id": f"squad_v2_{index}",
        "source": {"dataset": "squad_v2", "split": "train"},
        "prompt": {
            "question": f"What is question {index}?",
            "context": f"Context for question {index}.",
        },
        "reference": {"answerability": "ANSWERABLE"},
    }


def _generation(prompt_id, dataset_index, seed, status="SUCCESS"):
    return {
        "prompt_id": prompt_id,
        "sample_id": prompt_id,
        "dataset_index": dataset_index,
        "generation_index": seed - SEEDS[0],
        "seed": seed,
        "generation_id": f"{prompt_id}:{dataset_index}:{seed}",
        "generation_status": status,
        "model_name": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "device": "cpu",
        "thinking_mode": False,
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "max_new_tokens": 512,
        "generated_answer": f"answer {seed}",
    }


def _evaluation(generation_id, status, label, evaluator=EVALUATOR_MODEL, version=PROMPT_VERSION):
    return EvaluationRecord(
        schema_version=SCHEMA_VERSION,
        evaluator_model=evaluator,
        prompt_version=version,
        timestamp="2026-10-17T00:00:00+00:00",
        source_generation_id=generation_id,
        dataset_id="squad_v2",
        evaluation_status=status,
        retry_count=0,
        generation_label=label,
    ).validate()


class TestExpandedCohortSelection(unittest.TestCase):
    def _cohort(self, count):
        return [
            _make_scenario(index)
            for index in range(count)
        ]

    def test_selects_exactly_500_deterministically_and_preserves_existing(self):
        scenarios = self._cohort(520)
        existing = {
            "generations": [
                {
                    "prompt_id": scenarios[index]["sample_id"],
                    "dataset_index": index,
                }
                for index in range(31)
            ]
        }
        first = select_expanded_scenarios(scenarios, existing, 500, 20261017)
        second = select_expanded_scenarios(scenarios, existing, 500, 20261017)
        self.assertEqual(500, len(first))
        self.assertEqual(first, second)
        self.assertEqual(
            {scenarios[i]["sample_id"] for i in range(31)},
            {row["prompt_id"] for row in first if row["preserved_existing_prompt"]},
        )
        self.assertEqual(500, len({row["content_sha256"] for row in first}))
        self.assertEqual(500, len({row["prompt_id"] for row in first}))

    def test_duplicate_question_context_content_is_not_selected_twice(self):
        scenarios = self._cohort(30)
        duplicate = json.loads(json.dumps(scenarios[2]))
        duplicate["sample_id"] = "duplicate-id"
        scenarios.append(duplicate)
        selected = select_expanded_scenarios(
            scenarios, {"generations": []}, target_prompts=30, seed=4
        )
        self.assertEqual(30, len(selected))
        self.assertNotIn("duplicate-id", {row["prompt_id"] for row in selected})

    def test_fails_if_unique_source_is_smaller_than_requested(self):
        with self.assertRaisesRegex(ValueError, "only 20 are available"):
            select_expanded_scenarios(
                self._cohort(20), {"generations": []}, target_prompts=21
            )

    def test_batch_boundaries_are_stable_non_overlapping_100_prompt_slices(self):
        rows = [
            {"prompt_id": f"p{index}", "content_sha256": f"h{index}"}
            for index in range(500)
        ]
        selection = {
            "target_prompts": 500,
            "seeds": list(SEEDS),
            "selected_scenarios": rows,
        }
        batches = [batch_scenarios(selection, batch_id) for batch_id in range(1, 6)]
        self.assertEqual([BATCH_SIZE] * 5, [len(batch) for batch in batches])
        self.assertEqual(
            rows,
            [row for batch in batches for row in batch],
        )
        with self.assertRaisesRegex(ValueError, "between 1 and 5"):
            batch_scenarios(selection, 6)


class TestExpandedGenerationPersistence(unittest.TestCase):
    def test_resumes_from_sidecars_and_never_replaces_success(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            attempts = root / "generation_attempts"
            original_success = _generation("p0", 0, 1001)
            manifest = {"generations": [original_success]}
            failed = _generation("p0", 0, 1002, status="FAILED")
            failed["error"] = "interrupted"
            first_path = expanded._write_generation_attempt(attempts, failed)
            latest = expanded._latest_generation_records(manifest, attempts)
            self.assertEqual("FAILED", latest["p0:0:1002"]["generation_status"])

            success = _generation("p0", 0, 1002)
            retry_path = expanded._write_generation_attempt(attempts, success)
            self.assertTrue(retry_path.parent.name == "retries")
            latest = expanded._latest_generation_records(manifest, attempts)
            self.assertEqual("SUCCESS", latest["p0:0:1002"]["generation_status"])
            self.assertTrue(first_path.exists())
            self.assertTrue(retry_path.exists())
            self.assertEqual("SUCCESS", original_success["generation_status"])
            with self.assertRaisesRegex(ValueError, "successful generation"):
                expanded._write_generation_attempt(attempts, failed | {"seed": 1002})

    def test_generation_stage_reuses_completed_work_and_resumes_missing_prompts(self):
        class FakeTensor:
            shape = (1, 2)

        class FakeEncoding(dict):
            def to(self, _device):
                return self

        class FakeTokenizer:
            pad_token_id = 0
            eos_token_id = 0

            def __call__(self, _text, return_tensors=None, add_special_tokens=None):
                if return_tensors:
                    return FakeEncoding(input_ids=FakeTensor())
                return {"input_ids": [1, 2]}

        class FakeModel:
            def eval(self):
                return self

            def cpu(self):
                return self

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            canonical_path = root / "canonical.json"
            source_path = root / "source_generations.json"
            output_dir = root / "expanded"
            scenarios = [_make_scenario(0), _make_scenario(1)]
            canonical_path.write_text(json.dumps(scenarios), encoding="utf-8")
            original_records = [
                _generation(scenarios[0]["sample_id"], 0, seed)
                for seed in SEEDS
            ]
            source_manifest = {
                "pipeline": {
                    "seeds": list(SEEDS),
                    "num_prompts": 1,
                    "generations_per_prompt": 10,
                },
                "model": {
                    "name": MODEL_NAME,
                    "revision": MODEL_REVISION,
                    "device": "cpu",
                    "thinking_mode": False,
                    "decoding": {
                        "do_sample": True,
                        "temperature": 0.7,
                        "top_p": 0.8,
                        "top_k": 20,
                        "max_new_tokens": 512,
                    },
                },
                "generations": original_records,
                "scenarios": [{"dataset_index": 0, "sample_id": scenarios[0]["sample_id"]}],
                "summary": {},
            }
            source_path.write_text(json.dumps(source_manifest), encoding="utf-8")
            preserved_hidden = root / "data" / "production" / "hidden_states" / "hs_0.json"
            preserved_hidden.parent.mkdir(parents=True)
            preserved_hidden.write_text(json.dumps({
                "prompt_dataset_index": 0,
                "model_name": MODEL_NAME,
                "model_revision": MODEL_REVISION,
            }), encoding="utf-8")
            selection_artifact = create_selection_artifact(
                canonical_path, source_path, target_prompts=2, seed=7
            )
            (output_dir).mkdir(parents=True)
            (output_dir / "selection.json").write_text(
                json.dumps(selection_artifact), encoding="utf-8"
            )
            generated_ids = []
            hidden_calls = []
            fake_tokenizer = FakeTokenizer()
            fake_model = FakeModel()

            def generate(_model, _tokenizer, _encoded, seed, _pad):
                generated_ids.append(seed)
                return "generated", 1, 0.01, None

            def capture(_model, _tokenizer, _prompt):
                hidden_calls.append(True)
                return {
                    "hidden_size": 1,
                    "num_layers": 1,
                    "seq_len": 2,
                    "last_token": [[0.1]],
                    "mean_pooled": [[0.1]],
                }

            with (
                patch.object(expanded, "ROOT", root),
                patch.object(expanded, "CANONICAL_PATH", canonical_path),
                patch.object(expanded, "SOURCE_GENERATIONS_PATH", source_path),
                patch("run_cp4_2_pilot._load_model", return_value=(fake_tokenizer, fake_model)),
                patch("run_cp4_2_pilot.render_prompt", return_value="fake prompt"),
                patch("run_production.capture_hidden_states", side_effect=capture),
                patch("run_production.generate_answer", side_effect=generate),
            ):
                first = expanded.run_generation(
                    output_dir, max_new_prompts=1, batch_id=1
                )
                second = expanded.run_generation(
                    output_dir, max_new_prompts=1, batch_id=1
                )

            self.assertEqual(20, first["generation_count"])
            self.assertEqual("complete", first["status"])
            self.assertEqual("complete", second["status"])
            self.assertEqual(list(SEEDS), generated_ids)
            self.assertEqual(1, len(hidden_calls))
            self.assertEqual(
                original_records,
                [
                    record for record in load_json(output_dir / "generations.json")["generations"]
                    if record["prompt_id"] == scenarios[0]["sample_id"]
                ],
            )
            self.assertEqual(
                source_manifest["generations"],
                load_json(source_path)["generations"],
            )


class TestBatchEvaluationBoundary(unittest.TestCase):
    def test_evaluation_input_is_limited_to_completed_batch_prefix(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            selection = {
                "target_prompts": 500,
                "seeds": list(SEEDS),
                "selected_scenarios": [
                    {
                        "prompt_id": f"p{index}",
                        "dataset_index": index,
                        "content_sha256": f"h{index}",
                    }
                    for index in range(500)
                ],
            }
            selection_path = root / "selection.json"
            selection_path.write_text(json.dumps(selection), encoding="utf-8")
            generation_path = root / "generations.json"
            generations = []
            for row in selection["selected_scenarios"][:101]:
                for seed in SEEDS:
                    generations.append({
                        "prompt_id": row["prompt_id"],
                        "dataset_index": row["dataset_index"],
                        "generation_id": (
                            f"{row['prompt_id']}:{row['dataset_index']}:{seed}"
                        ),
                        "generation_status": "SUCCESS",
                    })
            generation_path.write_text(json.dumps({
                "cohort_selection": {
                    "selection_artifact_sha256": selection_digest(selection),
                },
                "generations": generations,
            }), encoding="utf-8")
            batch_dir = root / "batches"
            batch_dir.mkdir()
            (batch_dir / "batch_01.json").write_text(
                json.dumps({"status": "completed"}), encoding="utf-8"
            )

            with patch.object(eval_cp47, "EXPANDED_OUTPUT_DIR", root):
                evaluation_input = eval_cp47._batch_evaluation_input(
                    generation_path, batch_id=1
                )

            filtered = json.loads(evaluation_input.read_text(encoding="utf-8"))
            self.assertEqual(1000, len(filtered["generations"]))
            self.assertEqual({"p0", "p99"}, {
                filtered["generations"][0]["prompt_id"],
                filtered["generations"][-1]["prompt_id"],
            })
            self.assertNotIn(
                "p100",
                {record["prompt_id"] for record in filtered["generations"]},
            )


class TestExpandedTargetsAndPreflight(unittest.TestCase):
    def test_target_requires_ten_binary_same_version_evaluations(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = root / "generations.json"
            eval_dir = root / "evaluations"
            target_path = root / "targets.json"
            generations = []
            for index, prompt_id in enumerate(("complete", "incomplete")):
                for position, seed in enumerate(SEEDS):
                    generation = _generation(prompt_id, index, seed)
                    generations.append(generation)
                    if prompt_id == "complete":
                        label = (
                            GenerationLabel.UNRELIABLE
                            if position < 3 else GenerationLabel.RELIABLE
                        )
                        record = _evaluation(
                            generation["generation_id"],
                            EvaluationStatus.SUCCESS,
                            label,
                        )
                    elif position == 0:
                        record = _evaluation(
                            generation["generation_id"],
                            EvaluationStatus.SUCCESS,
                            GenerationLabel.RELIABLE,
                            evaluator="different-model",
                            version="OTHER",
                        )
                    else:
                        if position == 1:
                            record = _evaluation(
                                generation["generation_id"],
                                EvaluationStatus.MANUAL_REVIEW,
                                GenerationLabel.INADEQUATE,
                            )
                        else:
                            record = _evaluation(
                                generation["generation_id"],
                                EvaluationStatus.SUCCESS,
                                GenerationLabel.RELIABLE,
                            )
                    path = eval_dir / (
                        generation["generation_id"].replace(":", "_") + ".json"
                    )
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps(record.to_dict()), encoding="utf-8")
            manifest_path.write_text(
                json.dumps({"generations": generations}), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "mismatched provenance"):
                eval_runner.build_targets(
                    generation_path=manifest_path,
                    evaluations_dir=eval_dir,
                    targets_path=target_path,
                    evaluator_model=EVALUATOR_MODEL,
                    prompt_version=PROMPT_VERSION,
                )
            wrong_version_path = eval_dir / (
                f"incomplete_1_1001.json"
            )
            wrong_version_path.unlink()
            result = eval_runner.build_targets(
                generation_path=manifest_path,
                evaluations_dir=eval_dir,
                targets_path=target_path,
                evaluator_model=EVALUATOR_MODEL,
                prompt_version=PROMPT_VERSION,
            )
            rows = {row["prompt_id"]: row for row in result["targets"]}
            self.assertEqual(0.3, rows["complete"]["empirical_hallucination_risk"])
            self.assertTrue(rows["complete"]["target_eligible"])
            self.assertIsNone(rows["incomplete"]["empirical_hallucination_risk"])
            self.assertFalse(rows["incomplete"]["target_eligible"])
            self.assertEqual(1, rows["incomplete"]["n_manual_review"])

    def test_preflight_reports_missing_work_and_versioned_evaluations(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prod_dir = root / "production"
            output_dir = prod_dir / "expanded"
            old_hidden = prod_dir / "hidden_states" / "hs_0.json"
            old_hidden.parent.mkdir(parents=True)
            old_hidden.write_text(json.dumps({
                "prompt_dataset_index": 0,
                "model_name": MODEL_NAME,
                "model_revision": MODEL_REVISION,
            }), encoding="utf-8")
            existing_generations = [
                _generation("old-prompt", 0, seed) for seed in SEEDS
            ]
            eval_dir = prod_dir / "cp4_7" / "evaluations"
            for generation in existing_generations:
                path = eval_dir / (
                    generation["generation_id"].replace(":", "_") + ".json"
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                record = _evaluation(
                    generation["generation_id"],
                    EvaluationStatus.SUCCESS,
                    GenerationLabel.RELIABLE,
                )
                path.write_text(json.dumps(record.to_dict()), encoding="utf-8")
            selection = {
                "selected_scenarios": [
                    {"prompt_id": "old-prompt", "dataset_index": 0},
                    {"prompt_id": "new-prompt", "dataset_index": 1},
                ],
                "selection_seed": 13,
                "selection_method": "test",
                "duplicate_rule": "test",
                "canonical_scenarios_sha256": "hash",
                "canonical_unique_prompt_count": 2,
                "canonical_scenario_count": 2,
                "canonical_duplicate_content_count": 0,
                "existing_source_prompts": 1,
                "target_prompts": 2,
                "source_dataset": "squad_v2",
                "source_split": "train",
            }
            source_manifest = {"generations": existing_generations}
            report = build_preflight_report(
                selection, source_manifest, prod_dir, output_dir
            )
            self.assertEqual(20, report["generation"]["expected_total_generations"])
            self.assertEqual(10, report["generation"]["compatible_existing_generations"])
            self.assertEqual(10, report["generation"]["missing_generation_work"])
            self.assertEqual(10, report["evaluations"]["additional_evaluator_calls_expected"])
            self.assertEqual(
                {"gemini-3.5-flash-lite / CP4.7.0": {"SUCCESS": 10}},
                report["evaluations"]["existing_counts_by_evaluator_version_and_status"],
            )


class TestExpandedPredictorFeatures(unittest.TestCase):
    def test_features_use_only_pre_generation_fields(self):
        with tempfile.TemporaryDirectory() as temporary:
            hidden_dir = Path(temporary)
            hidden_path = hidden_dir / "hs_0.json"
            hidden_path.write_text(json.dumps({
                "prompt_dataset_index": 0,
                "model_name": MODEL_NAME,
                "model_revision": MODEL_REVISION,
            }), encoding="utf-8")
            selection = {
                "selected_scenarios": [
                    {
                        "prompt_id": "p0",
                        "dataset_index": 0,
                    }
                ]
            }
            scenario = _make_scenario(0)
            artifact = create_predictor_feature_artifact(
                selection,
                [scenario],
                hidden_dir,
                token_counter=lambda text: len(text.split()),
            )
            serialized = json.dumps(artifact["predictor_features"])
            for forbidden in (
                "generated_answer",
                "generation_label",
                "evaluation_status",
                "empirical_hallucination_risk",
                "claim_evaluations",
                "latency",
            ):
                self.assertNotIn(forbidden, serialized)
            self.assertEqual(1, len(artifact["predictor_features"]))

    def test_prompt_split_has_no_identity_overlap(self):
        from src.predictor.baselines import train_test_split_by_prompt

        ids = [f"prompt-{i}" for i in range(30)]
        train, test = train_test_split_by_prompt(ids, test_size=0.3, random_state=42)
        self.assertFalse(set(train) & set(test))
        self.assertEqual(set(ids), set(train) | set(test))
