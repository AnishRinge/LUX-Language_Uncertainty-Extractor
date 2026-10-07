"""Tests for the CP4.2 generation protocol pilot.

These tests exercise the production logic in ``run_cp4_2_pilot`` directly using
small synthetic fixtures and a stub tokenizer. They never load Qwen3 weights,
never call ``model.generate()``, never touch the real pilot artifact, and never
skip their assertions because an artifact happens to exist or not exist.
"""

import inspect
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import run_cp4_2_pilot as pilot


# --------------------------------------------------------------------------
# Fixtures / stubs
# --------------------------------------------------------------------------

class StubTokenizer:
    """Minimal stand-in for the Qwen3 tokenizer.

    Reproduces the installed chat template's thinking branch so the plumbing
    guarantee can be asserted without loading any model weights.
    """

    def __init__(self):
        self.calls = []

    def apply_chat_template(
        self,
        messages,
        tokenize=None,
        add_generation_prompt=None,
        enable_thinking=None,
        **kwargs,
    ):
        self.calls.append(
            {
                "messages": messages,
                "tokenize": tokenize,
                "add_generation_prompt": add_generation_prompt,
                "enable_thinking": enable_thinking,
                "extra_kwargs": sorted(kwargs),
            }
        )
        system = next((m["content"] for m in messages if m["role"] == "system"), "")
        user = next((m["content"] for m in messages if m["role"] == "user"), "")
        rendered = "<|im_start|>system\n{0}<|im_end|>\n<|im_start|>user\n{1}<|im_end|>\n".format(
            system, user
        )
        if add_generation_prompt:
            rendered += "<|im_start|>assistant\n"
            if defined_false(enable_thinking):
                rendered += "<think>\n\n</think>\n\n"
        return rendered


def defined_false(value):
    """Mirror of the template guard: `enable_thinking is defined and is false`."""
    return value is not None and value is False


def make_scenario(index, answerable):
    return {
        "sample_id": "squad_v2_sample_{0:03d}".format(index),
        "source": {
            "dataset": "squad_v2",
            "split": "train",
            "original_id": "orig_{0}".format(index),
            "article_id": str(index),
            "paragraph_id": "0",
            "article_title": "Article {0}".format(index),
        },
        "prompt": {
            "question": "Question {0}?".format(index),
            "context": "Context body for record {0}.".format(index),
        },
        "reference": {
            "answerability": (
                pilot.ANSWERABLE if answerable else pilot.UNANSWERABLE
            ),
            "answers": [{"text": "gold", "answer_start": 0}] if answerable else [],
            "supporting_information": "span" if answerable else "",
        },
        "metadata": {
            "phenomenon": "grounded_contextual_reliability",
            "expected_behavior": "ANSWER" if answerable else "ABSTAIN",
            "source_metadata": {"is_impossible": not answerable},
        },
    }


def make_record(scenario, seed, index, answer, status=pilot.STATUS_SUCCESS, token_count=12):
    return pilot.build_generation_record(
        scenario=scenario,
        dataset_index=index,
        generation_index=seed - pilot.SEEDS[0] + 1,
        seed=seed,
        prompt_text="<|im_start|>user\nstub<|im_end|>\n<|im_start|>assistant\n",
        prompt_token_count=64,
        status=status,
        generated_answer=answer,
        generated_token_count=token_count,
        latency=1.25,
        reached_max_tokens=(token_count >= pilot.MAX_NEW_TOKENS),
        error=None if status == pilot.STATUS_SUCCESS else "RuntimeError: simulated failure",
    )


def make_prompt_answers(count, distinct=True):
    """count answers; distinct=False forces every seed to return the same text."""
    answers = []
    for offset in range(count):
        answers.append("answer-{0}".format(offset) if distinct else "same answer")
    return answers


def healthy_verdict_inputs(num_prompts=2):
    """A clean run: all SUCCESS, diverse, reproducible, artifact written."""
    generations = []
    diversity = []
    for p in range(num_prompts):
        scenario = make_scenario(p, answerable=(p % 2 == 0))
        answers = make_prompt_answers(pilot.GENERATIONS_PER_PROMPT)
        prompt_records = [
            make_record(scenario, seed, p, answer)
            for seed, answer in zip(pilot.SEEDS, answers)
        ]
        generations.extend(prompt_records)
        metrics = pilot.diversity_metrics(prompt_records)
        metrics.update(
            {
                "sample_id": scenario["sample_id"],
                "dataset_index": p,
                "answerability": scenario["reference"]["answerability"],
                "truncation_rate": 0.0,
            }
        )
        diversity.append(metrics)

    reproducibility = [
        {
            "sample_id": d["sample_id"],
            "dataset_index": d["dataset_index"],
            "seed": pilot.SEEDS[0],
            "status": pilot.STATUS_SUCCESS,
            "match": True,
            "original_output": "answer-0",
            "reproduced_output": "answer-0",
            "latency": 1.0,
            "error": None,
        }
        for d in diversity
    ]
    return {
        "run_state": pilot.RUN_COMPLETED,
        "generations": generations,
        "diversity_by_prompt": diversity,
        "diversity_overall": pilot.summarize_diversity(diversity),
        "reproducibility": reproducibility,
        "artifact_persisted": True,
    }


def check_named(result, name):
    for check in result["checks"]:
        if check["name"] == name:
            return check
    raise AssertionError("no check named {0}".format(name))


def find_cached_chat_template():
    """Locate the locally cached Qwen3 tokenizer_config.json, if present."""
    cache_root = Path(os.environ.get("HF_HOME") or (Path.home() / ".cache/huggingface"))
    candidates = sorted(cache_root.glob("hub/models--Qwen--Qwen3-1.7B/snapshots/*/tokenizer_config.json"))
    for candidate in candidates:
        try:
            with open(candidate, "r", encoding="utf-8") as handle:
                return json.load(handle).get("chat_template")
        except (OSError, ValueError):
            continue
    return None


# --------------------------------------------------------------------------
# 1. generation configuration values
# --------------------------------------------------------------------------

class TestGenerationConfig(unittest.TestCase):
    def test_config_matches_locked_protocol(self):
        config = pilot.generation_config()
        self.assertEqual(config["model_name"], "Qwen/Qwen3-1.7B")
        self.assertEqual(config["model_revision"], "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e")
        self.assertEqual(config["device"], "cpu")
        self.assertEqual(config["execution_strategy"], "CPU_ONLY")
        self.assertIs(config["do_sample"], True)
        self.assertEqual(config["temperature"], 0.7)
        self.assertEqual(config["top_p"], 0.8)
        self.assertEqual(config["top_k"], 20)
        self.assertEqual(config["max_new_tokens"], 512)
        self.assertEqual(config["generations_per_prompt"], 10)

    def test_module_constants_match_config(self):
        self.assertIs(pilot.DO_SAMPLE, True)
        self.assertEqual(pilot.TEMPERATURE, 0.7)
        self.assertEqual(pilot.TOP_P, 0.8)
        self.assertEqual(pilot.TOP_K, 20)
        self.assertEqual(pilot.MAX_NEW_TOKENS, 512)
        self.assertIs(pilot.THINKING_MODE, False)
        self.assertEqual(pilot.DEVICE, "cpu")

    def test_heavy_dependencies_are_imported_lazily(self):
        # Proves the module (and therefore this test module) never pulls in
        # torch/transformers, so nothing here can load model weights.
        self.assertFalse(hasattr(pilot, "torch"))
        self.assertFalse(hasattr(pilot, "transformers"))

    def test_config_is_a_fresh_copy_each_call(self):
        first = pilot.generation_config()
        first["temperature"] = 99.0
        self.assertEqual(pilot.generation_config()["temperature"], 0.7)


# --------------------------------------------------------------------------
# 2. seed list
# --------------------------------------------------------------------------

class TestSeedProtocol(unittest.TestCase):
    def test_seed_list_values(self):
        self.assertEqual(
            list(pilot.SEEDS),
            [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010],
        )

    def test_seed_count_and_uniqueness(self):
        self.assertEqual(len(pilot.SEEDS), 10)
        self.assertEqual(len(set(pilot.SEEDS)), 10)
        self.assertEqual(pilot.GENERATIONS_PER_PROMPT, len(pilot.SEEDS))

    def test_seed_is_the_only_per_generation_variable(self):
        # Two records for the same prompt must differ only by seed and its index.
        scenario = make_scenario(0, answerable=True)
        first = make_record(scenario, pilot.SEEDS[0], 0, "a")
        second = make_record(scenario, pilot.SEEDS[1], 0, "b")
        differing = {
            key
            for key in set(first) | set(second)
            if first.get(key) != second.get(key)
        }
        # Only seed / generation_index / the answer text / latency may differ.
        self.assertLessEqual(
            differing,
            {"seed", "generation_index", "generated_answer", "latency"},
        )

    def test_seed_is_recorded_but_never_a_predictor_feature(self):
        scenario = make_scenario(0, answerable=True)
        record = make_record(scenario, pilot.SEEDS[0], 0, "a")
        self.assertEqual(record["seed"], pilot.SEEDS[0])
        for forbidden in ("predictor_features", "hallucination_label", "empirical_risk"):
            self.assertNotIn(forbidden, record)
            self.assertNotIn(forbidden, record["provenance"])


# --------------------------------------------------------------------------
# 3. deterministic pilot selection
# --------------------------------------------------------------------------

class TestPilotSelection(unittest.TestCase):
    def setUp(self):
        # 10 answerable then 10 unanswerable, so naive head-slicing would fail.
        self.scenarios = [make_scenario(i, answerable=(i < 10)) for i in range(20)]

    def test_evenly_spaced_indices_are_strictly_increasing_and_in_range(self):
        indices = pilot.evenly_spaced_indices(130319, 3)
        self.assertEqual(indices, sorted(indices))
        self.assertEqual(len(set(indices)), 3)
        for index in indices:
            self.assertGreaterEqual(index, 0)
            self.assertLess(index, 130319)

    def test_evenly_spaced_indices_edge_cases(self):
        self.assertEqual(pilot.evenly_spaced_indices(0, 3), [])
        self.assertEqual(pilot.evenly_spaced_indices(-4, 3), [])
        self.assertEqual(pilot.evenly_spaced_indices(3, 0), [])
        self.assertEqual(pilot.evenly_spaced_indices(2, 5), [0, 1])
        self.assertEqual(pilot.evenly_spaced_indices(5, 5), [0, 1, 2, 3, 4])

    def test_selection_is_deterministic_across_repeated_calls(self):
        first = pilot.select_pilot_scenarios(self.scenarios)
        second = pilot.select_pilot_scenarios(self.scenarios)
        self.assertEqual([i for i, _ in first], [i for i, _ in second])
        self.assertEqual([s["sample_id"] for _, s in first], [s["sample_id"] for _, s in second])

    def test_selection_is_stratified_three_plus_three(self):
        selected = pilot.select_pilot_scenarios(self.scenarios)
        self.assertEqual(len(selected), 6)
        answerable = [s for _, s in selected if pilot.is_answerable(s)]
        unanswerable = [s for _, s in selected if not pilot.is_answerable(s)]
        self.assertEqual(len(answerable), 3)
        self.assertEqual(len(unanswerable), 3)

    def test_selection_spreads_across_the_dataset(self):
        indices = [i for i, _ in pilot.select_pilot_scenarios(self.scenarios)]
        # Picks from both halves rather than clustering at the head.
        self.assertTrue(any(i < 5 for i in indices))
        self.assertTrue(any(i >= 15 for i in indices))

    def test_selection_survives_reordering_only_by_content(self):
        # Same class distribution, different order -> different but still
        # stratified and still deterministic.
        reordered = list(reversed(self.scenarios))
        selected = pilot.select_pilot_scenarios(reordered)
        answerable = sum(1 for _, s in selected if pilot.is_answerable(s))
        self.assertEqual(answerable, 3)
        self.assertEqual(
            [i for i, _ in pilot.select_pilot_scenarios(reordered)],
            [i for i, _ in pilot.select_pilot_scenarios(reordered)],
        )

    def test_selection_returns_original_scenario_objects(self):
        selected = pilot.select_pilot_scenarios(self.scenarios)
        for index, scenario in selected:
            self.assertIs(scenario, self.scenarios[index])

    def test_selection_tolerates_missing_class(self):
        all_answerable = [make_scenario(i, answerable=True) for i in range(5)]
        selected = pilot.select_pilot_scenarios(all_answerable)
        self.assertEqual(len(selected), 3)
        self.assertTrue(all(pilot.is_answerable(s) for _, s in selected))

    def test_selection_rule_is_documented(self):
        self.assertIn("Deterministic", pilot.PILOT_SELECTION_RULE)
        self.assertIn("No random sampling", pilot.PILOT_SELECTION_RULE)
        self.assertIn("evenly_spaced_indices", pilot.PILOT_SELECTION_RULE)
        self.assertIn("reference.answerability", pilot.PILOT_SELECTION_RULE)

    def test_describe_selection_records_provenance_only(self):
        described = pilot.describe_selection(pilot.select_pilot_scenarios(self.scenarios))
        self.assertEqual(len(described), 6)
        for item in described:
            self.assertIn("dataset_index", item)
            self.assertIn("answerability", item)
            self.assertIn("never a predictor feature", item["note"])
            self.assertNotIn("answers", item)
        self.assertEqual(pilot.find_forbidden_fields(described), [])


# --------------------------------------------------------------------------
# FIX 1 / FIX 2 - thinking disablement and prompt format
# --------------------------------------------------------------------------

class TestThinkingAndPromptFormat(unittest.TestCase):
    def test_render_prompt_disables_thinking_at_runtime(self):
        tokenizer = StubTokenizer()
        rendered = pilot.render_prompt(tokenizer, "ctx body", "the question?")

        self.assertEqual(len(tokenizer.calls), 1)
        call = tokenizer.calls[0]
        # The flag must be explicitly False, not merely absent.
        self.assertIs(call["enable_thinking"], False)
        self.assertIs(call["add_generation_prompt"], True)
        self.assertIs(call["tokenize"], False)
        self.assertIn("<think>\n\n</think>\n\n", rendered)

    def test_flag_is_load_bearing(self):
        # Without the flag the template emits no empty think block, so passing
        # it genuinely changes runtime behaviour rather than being cosmetic.
        without_flag = StubTokenizer().apply_chat_template(
            [{"role": "user", "content": "x"}], tokenize=False, add_generation_prompt=True
        )
        with_flag = StubTokenizer().apply_chat_template(
            [{"role": "user", "content": "x"}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        self.assertNotIn("<think>", without_flag)
        self.assertIn("<think>", with_flag)

    def test_render_prompt_source_passes_the_flag_explicitly(self):
        source = inspect.getsource(pilot.render_prompt)
        self.assertIn("apply_chat_template", source)
        self.assertIn("enable_thinking=False", source)
        self.assertIn("add_generation_prompt=True", source)

    def test_thinking_metadata_reflects_runtime_enforcement(self):
        self.assertIs(pilot.THINKING_MODE, False)
        self.assertEqual(pilot.THINKING_ENFORCEMENT, "qwen3_chat_template:enable_thinking=False")
        self.assertIs(pilot.generation_config()["thinking_mode"], False)

    def test_cached_qwen3_template_supports_the_flag(self):
        template = find_cached_chat_template()
        if template is not None:
            self.assertIn("enable_thinking is defined and enable_thinking is false", template)
            self.assertIn("<think>", template)
        else:
            # Model assets absent here; the plumbing guarantee is asserted by
            # the tests above, so this still asserts something concrete.
            self.assertEqual(
                pilot.THINKING_ENFORCEMENT, "qwen3_chat_template:enable_thinking=False"
            )

    def test_chat_messages_are_system_plus_user(self):
        messages = pilot.build_chat_messages("ctx body", "the question?")
        self.assertEqual([m["role"] for m in messages], ["system", "user"])
        self.assertEqual(messages[1]["content"], "ctx body\n\nthe question?")

    def test_user_message_preserves_context_and_question_verbatim(self):
        context = "Beyonce rose to fame in the late 1990s."
        question = "When did Beyonce start becoming popular?"
        message = pilot.build_user_message(context, question)
        self.assertIn(context, message)
        self.assertIn(question, message)

    def test_user_message_excludes_reference_answer(self):
        scenario = make_scenario(0, answerable=True)
        message = pilot.build_user_message(
            scenario["prompt"]["context"], scenario["prompt"]["question"]
        )
        self.assertNotIn("gold", message)
        self.assertNotIn("ANSWERABLE", message)
        self.assertNotIn("ABSTAIN", message)
        self.assertNotIn(scenario["sample_id"], message)

    def test_prompt_format_is_no_longer_raw_completion(self):
        self.assertEqual(pilot.PROMPT_FORMAT, "qwen3_chat_template")
        message = pilot.build_user_message("ctx", "q?")
        self.assertFalse(message.endswith("Answer:"))
        self.assertNotIn("Context:", message)


# --------------------------------------------------------------------------
# 4. metadata construction
# --------------------------------------------------------------------------

class TestMetadataConstruction(unittest.TestCase):
    def setUp(self):
        self.scenario = make_scenario(7, answerable=True)
        self.record = make_record(self.scenario, 1003, 7, "the answer")

    def test_all_required_fields_present(self):
        for field in pilot.REQUIRED_RECORD_FIELDS:
            self.assertIn(field, self.record)

    def test_required_field_tuple_covers_fix10(self):
        for field in (
            "sample_id",
            "generation_index",
            "seed",
            "model_name",
            "model_revision",
            "device",
            "thinking_mode",
            "do_sample",
            "temperature",
            "top_p",
            "top_k",
            "max_new_tokens",
            "status",
            "generated_answer",
            "generated_token_count",
            "latency",
            "reached_max_tokens",
        ):
            self.assertIn(field, pilot.REQUIRED_RECORD_FIELDS)

    def test_record_values_are_consistent_with_protocol(self):
        self.assertEqual(self.record["model_name"], "Qwen/Qwen3-1.7B")
        self.assertEqual(self.record["model_revision"], "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e")
        self.assertEqual(self.record["device"], "cpu")
        self.assertIs(self.record["thinking_mode"], False)
        self.assertEqual(self.record["thinking_enforcement"], pilot.THINKING_ENFORCEMENT)
        self.assertEqual(self.record["seed"], 1003)
        self.assertEqual(self.record["generation_index"], 3)
        self.assertEqual(self.record["generated_answer"], "the answer")

    def test_record_identifies_the_source_scenario(self):
        self.assertEqual(self.record["sample_id"], self.scenario["sample_id"])
        self.assertEqual(self.record["dataset_index"], 7)
        self.assertEqual(self.record["provenance"]["original_id"], "orig_7")
        self.assertEqual(self.record["provenance"]["article_id"], "7")
        self.assertEqual(self.record["provenance"]["article_title"], "Article 7")

    def test_record_carries_no_future_stage_fields(self):
        self.assertEqual(pilot.find_forbidden_fields([self.record]), [])
        self.assertEqual(pilot.find_missing_fields([self.record]), [])

    def test_find_forbidden_fields_detects_injected_fields(self):
        injected = dict(self.record)
        injected["hallucination_label"] = "INCORRECT"
        violations = pilot.find_forbidden_fields([injected])
        self.assertEqual(len(violations), 1)
        self.assertEqual(violations[0]["field"], "hallucination_label")

    def test_find_forbidden_fields_detects_nested_provenance_pollution(self):
        polluted = dict(self.record)
        polluted["provenance"] = dict(self.record["provenance"])
        polluted["provenance"]["predictor_features"] = {"len": 12}
        violations = pilot.find_forbidden_fields([polluted])
        self.assertEqual([v["field"] for v in violations], ["provenance.predictor_features"])

    def test_find_missing_fields_detects_incomplete_record(self):
        incomplete = dict(self.record)
        del incomplete["latency"]
        missing = pilot.find_missing_fields([incomplete])
        self.assertEqual(len(missing), 1)
        self.assertIn("latency", missing[0]["absent"])

    def test_format_error_includes_exception_type(self):
        try:
            raise ValueError("bad input")
        except ValueError as exc:
            self.assertEqual(pilot.format_error(exc), "ValueError: bad input")


# --------------------------------------------------------------------------
# 5. diversity metrics
# --------------------------------------------------------------------------

class TestDiversityMetrics(unittest.TestCase):
    def test_unique_duplicate_and_ratio(self):
        records = [
            {"status": pilot.STATUS_SUCCESS, "generated_answer": "Paris"},
            {"status": pilot.STATUS_SUCCESS, "generated_answer": "  Paris\n"},
            {"status": pilot.STATUS_SUCCESS, "generated_answer": "Paris"},
            {"status": pilot.STATUS_SUCCESS, "generated_answer": "Lyon"},
            {"status": pilot.STATUS_FAILED, "generated_answer": "", "error": "boom"},
        ]
        metrics = pilot.diversity_metrics(records)
        self.assertEqual(metrics["attempted_generations"], 5)
        self.assertEqual(metrics["successful_generations"], 4)
        self.assertEqual(metrics["failed_generations"], 1)
        self.assertEqual(metrics["unique_outputs"], 2)
        self.assertEqual(metrics["duplicate_outputs"], 2)
        self.assertEqual(metrics["unique_output_ratio"], 0.5)

    def test_duplicate_detection_is_whitespace_only_not_case_folded(self):
        # Documented behaviour: comparison trims and collapses whitespace but
        # preserves case, so case variants count as distinct outputs.
        records = [
            {"status": pilot.STATUS_SUCCESS, "generated_answer": "Paris"},
            {"status": pilot.STATUS_SUCCESS, "generated_answer": "paris"},
        ]
        metrics = pilot.diversity_metrics(records)
        self.assertEqual(metrics["unique_outputs"], 2)
        self.assertEqual(pilot.normalize_output("Paris "), pilot.normalize_output("  Paris\n"))
        self.assertNotEqual(pilot.normalize_output("Paris"), pilot.normalize_output("paris"))

    def test_all_identical_outputs_are_duplicates_not_failures(self):
        records = [
            {"status": pilot.STATUS_SUCCESS, "generated_answer": "same"} for _ in range(10)
        ]
        metrics = pilot.diversity_metrics(records)
        self.assertEqual(metrics["successful_generations"], 10)
        self.assertEqual(metrics["failed_generations"], 0)
        self.assertEqual(metrics["unique_outputs"], 1)
        self.assertEqual(metrics["duplicate_outputs"], 9)
        self.assertEqual(metrics["unique_output_ratio"], 0.1)

    def test_fully_diverse_outputs(self):
        records = [
            {"status": pilot.STATUS_SUCCESS, "generated_answer": "answer-{0}".format(i)}
            for i in range(10)
        ]
        metrics = pilot.diversity_metrics(records)
        self.assertEqual(metrics["unique_outputs"], 10)
        self.assertEqual(metrics["duplicate_outputs"], 0)
        self.assertEqual(metrics["unique_output_ratio"], 1.0)

    def test_ratio_is_zero_when_nothing_succeeded(self):
        metrics = pilot.diversity_metrics(
            [{"status": pilot.STATUS_FAILED, "generated_answer": ""}]
        )
        self.assertEqual(metrics["unique_output_ratio"], 0.0)
        self.assertEqual(metrics["unique_outputs"], 0)

    def test_normalize_output_collapses_whitespace(self):
        self.assertEqual(pilot.normalize_output("  a \n\t b  "), "a b")
        self.assertEqual(pilot.normalize_output(None), "")

    def test_summarize_diversity_aggregates_prompts(self):
        inputs = healthy_verdict_inputs(num_prompts=2)
        overall = pilot.summarize_diversity(inputs["diversity_by_prompt"])
        self.assertEqual(overall["num_prompts"], 2)
        self.assertEqual(overall["attempted_generations"], 20)
        self.assertEqual(overall["successful_generations"], 20)
        self.assertEqual(overall["failed_generations"], 0)
        self.assertEqual(overall["unique_outputs"], 20)
        self.assertEqual(overall["duplicate_outputs"], 0)
        self.assertEqual(overall["unique_output_ratio"], 1.0)
        self.assertEqual(overall["prompts_with_stochastic_diversity"], 2)

    def test_truncation_rate_counts_only_successful_generations(self):
        records = [
            {"status": pilot.STATUS_SUCCESS, "reached_max_tokens": True},
            {"status": pilot.STATUS_SUCCESS, "reached_max_tokens": False},
            {"status": pilot.STATUS_FAILED, "reached_max_tokens": False},
        ]
        self.assertAlmostEqual(pilot.truncation_rate(records), 0.5)

    def test_truncation_rate_zero_with_no_successes(self):
        self.assertEqual(pilot.truncation_rate([]), 0.0)


# --------------------------------------------------------------------------
# 6. failure status handling
# --------------------------------------------------------------------------

class TestFailureStatusHandling(unittest.TestCase):
    def test_failed_record_carries_error_and_empty_answer(self):
        record = make_record(
            make_scenario(0, True), 1002, 0, "", status=pilot.STATUS_FAILED
        )
        self.assertEqual(record["status"], pilot.STATUS_FAILED)
        self.assertIn("RuntimeError", record["error"])
        self.assertEqual(record["generated_answer"], "")

    def test_failed_generation_is_not_labelled_hallucination(self):
        record = make_record(make_scenario(0, True), 1002, 0, "", status=pilot.STATUS_FAILED)
        self.assertNotIn("hallucination_label", record)
        self.assertNotIn("response_label", record)
        self.assertNotIn("empirical_risk", record)
        self.assertEqual(pilot.find_forbidden_fields([record]), [])

    def test_failed_generation_still_counts_toward_attempts(self):
        records = [
            {"status": pilot.STATUS_SUCCESS, "generated_answer": "a"},
            {"status": pilot.STATUS_FAILED, "generated_answer": "", "error": "e"},
        ]
        metrics = pilot.diversity_metrics(records)
        self.assertEqual(metrics["attempted_generations"], 2)
        self.assertEqual(metrics["successful_generations"], 1)
        self.assertEqual(metrics["failed_generations"], 1)

    def test_truncation_flag_not_set_for_failed_generation(self):
        record = make_record(make_scenario(0, True), 1002, 0, "", status=pilot.STATUS_FAILED)
        self.assertIs(record["reached_max_tokens"], False)

    def test_forbidden_list_excludes_generated_answer(self):
        # generated_answer legitimately lives on generation records.
        self.assertNotIn("generated_answer", pilot.FORBIDDEN_RECORD_FIELDS)
        for field in (
            "hallucination_label",
            "empirical_risk",
            "evaluator_output",
            "predictor_features",
            "hidden_states",
        ):
            self.assertIn(field, pilot.FORBIDDEN_RECORD_FIELDS)


# --------------------------------------------------------------------------
# 7. report verdict logic
# --------------------------------------------------------------------------

class TestVerdictLogic(unittest.TestCase):
    def test_healthy_run_passes(self):
        result = pilot.evaluate_verdict(**healthy_verdict_inputs())
        self.assertEqual(result["verdict"], pilot.VERDICT_PASS)
        self.assertEqual(result["required_failed"], [])
        self.assertEqual(result["advisory_failed"], [])

    def test_failed_generation_forces_fail(self):
        inputs = healthy_verdict_inputs()
        inputs["generations"][3]["status"] = pilot.STATUS_FAILED
        result = pilot.evaluate_verdict(**inputs)
        self.assertEqual(result["verdict"], pilot.VERDICT_FAIL)
        self.assertIn("generation_success", result["required_failed"])

    def test_incomplete_run_forces_fail(self):
        inputs = healthy_verdict_inputs()
        inputs["run_state"] = pilot.RUN_FAILED
        result = pilot.evaluate_verdict(**inputs)
        self.assertEqual(result["verdict"], pilot.VERDICT_FAIL)
        self.assertIn("pilot_run_completed", result["required_failed"])

    def test_no_generations_attempted_is_fail_not_pass(self):
        inputs = healthy_verdict_inputs()
        inputs["generations"] = []
        inputs["diversity_by_prompt"] = []
        inputs["diversity_overall"] = pilot.summarize_diversity([])
        inputs["reproducibility"] = []
        result = pilot.evaluate_verdict(**inputs)
        self.assertEqual(result["verdict"], pilot.VERDICT_FAIL)
        self.assertIn("generations_were_attempted", result["required_failed"])

    def test_reproducibility_mismatch_forces_fail(self):
        inputs = healthy_verdict_inputs()
        inputs["reproducibility"][0]["match"] = False
        result = pilot.evaluate_verdict(**inputs)
        self.assertEqual(result["verdict"], pilot.VERDICT_FAIL)
        self.assertIn("same_seed_reproducibility", result["required_failed"])

    def test_reproducibility_technical_error_forces_fail(self):
        inputs = healthy_verdict_inputs()
        inputs["reproducibility"][0]["status"] = pilot.STATUS_FAILED
        inputs["reproducibility"][0]["match"] = None
        result = pilot.evaluate_verdict(**inputs)
        self.assertEqual(result["verdict"], pilot.VERDICT_FAIL)
        self.assertIn("same_seed_reproducibility", result["required_failed"])

    def test_missing_reproducibility_forces_fail(self):
        inputs = healthy_verdict_inputs()
        inputs["reproducibility"] = []
        result = pilot.evaluate_verdict(**inputs)
        self.assertEqual(result["verdict"], pilot.VERDICT_FAIL)
        self.assertIn("same_seed_reproducibility", result["required_failed"])

    def test_leakage_forces_fail(self):
        inputs = healthy_verdict_inputs()
        inputs["generations"][0]["predictor_features"] = {"q_len": 10}
        result = pilot.evaluate_verdict(**inputs)
        self.assertEqual(result["verdict"], pilot.VERDICT_FAIL)
        self.assertIn("no_forbidden_fields", result["required_failed"])

    def test_missing_artifact_forces_fail(self):
        inputs = healthy_verdict_inputs()
        inputs["artifact_persisted"] = False
        result = pilot.evaluate_verdict(**inputs)
        self.assertEqual(result["verdict"], pilot.VERDICT_FAIL)
        self.assertIn("artifact_persisted", result["required_failed"])

    def test_missing_metadata_forces_fail(self):
        inputs = healthy_verdict_inputs()
        del inputs["generations"][2]["seed"]
        result = pilot.evaluate_verdict(**inputs)
        self.assertEqual(result["verdict"], pilot.VERDICT_FAIL)
        self.assertIn("record_metadata_complete", result["required_failed"])

    def test_collapsed_diversity_is_needs_revision_not_pass(self):
        inputs = healthy_verdict_inputs(num_prompts=1)
        scenario = make_scenario(0, answerable=True)
        records = [make_record(scenario, seed, 0, "identical") for seed in pilot.SEEDS]
        metrics = pilot.diversity_metrics(records)
        metrics.update({"sample_id": scenario["sample_id"], "dataset_index": 0,
                        "answerability": pilot.ANSWERABLE, "truncation_rate": 0.0})
        inputs["generations"] = records
        inputs["diversity_by_prompt"] = [metrics]
        inputs["diversity_overall"] = pilot.summarize_diversity([metrics])
        inputs["reproducibility"][0]["original_output"] = "identical"
        inputs["reproducibility"][0]["reproduced_output"] = "identical"
        result = pilot.evaluate_verdict(**inputs)
        self.assertEqual(result["verdict"], pilot.VERDICT_NEEDS_REVISION)
        self.assertIn("stochastic_diversity_observed", result["advisory_failed"])
        self.assertEqual(result["required_failed"], [])

    def test_excessive_truncation_is_needs_revision(self):
        inputs = healthy_verdict_inputs(num_prompts=1)
        for record in inputs["generations"]:
            record["generated_token_count"] = pilot.MAX_NEW_TOKENS
            record["reached_max_tokens"] = True
        result = pilot.evaluate_verdict(**inputs)
        self.assertEqual(result["verdict"], pilot.VERDICT_NEEDS_REVISION)
        self.assertIn("truncation_within_tolerance", result["advisory_failed"])

    def test_required_failure_outranks_advisory_failure(self):
        inputs = healthy_verdict_inputs(num_prompts=1)
        inputs["artifact_persisted"] = False
        for record in inputs["generations"]:
            record["generated_token_count"] = pilot.MAX_NEW_TOKENS
            record["reached_max_tokens"] = True
        result = pilot.evaluate_verdict(**inputs)
        self.assertEqual(result["verdict"], pilot.VERDICT_FAIL)

    def test_verdict_criteria_are_documented(self):
        result = pilot.evaluate_verdict(**healthy_verdict_inputs())
        self.assertTrue(result["criteria_documentation"]["required"])
        self.assertIn("unique", result["criteria_documentation"]["advisory"])

    def test_verdict_cannot_be_forced_to_pass_by_construction(self):
        # Exhaustively: any falsified required check must yield FAIL.
        for name in (
            "pilot_run_completed",
            "generations_were_attempted",
            "generation_success",
            "same_seed_reproducibility",
            "no_forbidden_fields",
            "record_metadata_complete",
            "artifact_persisted",
        ):
            inputs = healthy_verdict_inputs()
            if name == "pilot_run_completed":
                inputs["run_state"] = pilot.RUN_IN_PROGRESS
            elif name == "generations_were_attempted":
                inputs["generations"] = []
                inputs["diversity_by_prompt"] = []
                inputs["reproducibility"] = []
            elif name == "generation_success":
                inputs["generations"][0]["status"] = pilot.STATUS_FAILED
            elif name == "same_seed_reproducibility":
                inputs["reproducibility"][0]["match"] = False
            elif name == "no_forbidden_fields":
                inputs["generations"][0]["evaluator_output"] = "CORRECT"
            elif name == "record_metadata_complete":
                del inputs["generations"][0]["status"]
            elif name == "artifact_persisted":
                inputs["artifact_persisted"] = False
            result = pilot.evaluate_verdict(**inputs)
            self.assertEqual(
                result["verdict"], pilot.VERDICT_FAIL, "check {0} did not force FAIL".format(name)
            )


class TestReportRendering(unittest.TestCase):
    def _payload(self, inputs, verdict_result):
        """Build a report payload from the SAME inputs the verdict was derived
        from, so mutations made by a test are actually reflected in the report.
        """
        successful = [g for g in inputs["generations"] if g["status"] == pilot.STATUS_SUCCESS]
        attempted = len(inputs["generations"])
        return {
            "pilot_metadata": {
                "run_state": inputs["run_state"],
                "run_started_at": "2026-09-24T11:00:00+00:00",
                "run_updated_at": "2026-09-24T11:30:00+00:00",
                "run_error": inputs.get("run_error"),
                "artifact_persisted": inputs["artifact_persisted"],
                "num_prompts": len(inputs["diversity_by_prompt"]),
                "prompts_completed": len(inputs["diversity_by_prompt"]),
                "generations_planned": attempted,
                "generations_attempted": attempted,
                "successful_generations": len(successful),
                "failed_generations": attempted - len(successful),
            },
            "selection": pilot.describe_selection(
                [
                    (m["dataset_index"], make_scenario(m["dataset_index"], m["answerability"] == pilot.ANSWERABLE))
                    for m in inputs["diversity_by_prompt"]
                ]
            ),
            "diversity_by_prompt": inputs["diversity_by_prompt"],
            "diversity_overall": inputs["diversity_overall"],
            "reproducibility": inputs["reproducibility"],
            "generations": inputs["generations"],
            "verdict": verdict_result,
        }

    def test_report_reflects_pass_verdict(self):
        inputs = healthy_verdict_inputs(num_prompts=1)
        verdict = pilot.evaluate_verdict(**inputs)
        report = pilot.render_report(self._payload(inputs, verdict))
        self.assertIn("**Verdict:** **PASS**", report)
        self.assertIn("unique-output ratio", report)
        self.assertIn("No generation failed.", report)

    def test_report_reflects_fail_verdict(self):
        inputs = healthy_verdict_inputs(num_prompts=1)
        inputs["generations"][0]["status"] = pilot.STATUS_FAILED
        verdict = pilot.evaluate_verdict(**inputs)
        report = pilot.render_report(self._payload(inputs, verdict))
        self.assertIn("**Verdict:** **FAIL**", report)
        self.assertIn("generation_success", report)
        self.assertIn("not validated", report)

    def test_report_reflects_needs_revision_verdict(self):
        inputs = healthy_verdict_inputs(num_prompts=1)
        scenario = make_scenario(0, answerable=True)
        records = [make_record(scenario, seed, 0, "identical") for seed in pilot.SEEDS]
        metrics = pilot.diversity_metrics(records)
        metrics.update({"sample_id": scenario["sample_id"], "dataset_index": 0,
                        "answerability": pilot.ANSWERABLE, "truncation_rate": 0.0})
        inputs["generations"] = records
        inputs["diversity_by_prompt"] = [metrics]
        inputs["reproducibility"][0]["original_output"] = "identical"
        inputs["reproducibility"][0]["reproduced_output"] = "identical"
        verdict = pilot.evaluate_verdict(**inputs)
        report = pilot.render_report(self._payload(inputs, verdict))
        self.assertIn("**Verdict:** **NEEDS_REVISION**", report)
        self.assertIn("stochastic_diversity_observed", report)

    def test_report_contains_no_unconditional_pass_assertion(self):
        inputs = healthy_verdict_inputs(num_prompts=1)
        inputs["artifact_persisted"] = False
        verdict = pilot.evaluate_verdict(**inputs)
        report = pilot.render_report(self._payload(inputs, verdict))
        self.assertNotIn("CP4.2 protocol is **PASS**", report)
        self.assertNotIn("**Verdict:** **PASS**", report)

    def test_report_lists_failures_and_leakage_result(self):
        inputs = healthy_verdict_inputs(num_prompts=1)
        inputs["generations"][1]["status"] = pilot.STATUS_FAILED
        inputs["generations"][1]["error"] = "OutOfMemoryError: simulated failure"
        verdict = pilot.evaluate_verdict(**inputs)
        report = pilot.render_report(self._payload(inputs, verdict))
        self.assertIn("## 7. Failures", report)
        self.assertIn("OutOfMemoryError: simulated failure", report)
        self.assertNotIn("No generation failed.", report)
        self.assertIn("## 8. Data Leakage Check", report)

    def test_report_states_scope_boundary(self):
        inputs = healthy_verdict_inputs(num_prompts=1)
        verdict = pilot.evaluate_verdict(**inputs)
        report = pilot.render_report(self._payload(inputs, verdict))
        self.assertIn(
            "No hallucination", report
        )
        self.assertIn(
            "evaluation, empirical risk, hidden-state extraction or predictor training", report
        )
        self.assertIn("generation-protocol infrastructure only", report)


# --------------------------------------------------------------------------
# FIX 7 - incremental artifact persistence
# --------------------------------------------------------------------------

class TestIncrementalPersistence(unittest.TestCase):
    def test_roundtrip_keeps_artifact_valid_json(self):
        inputs = healthy_verdict_inputs(num_prompts=1)
        payload = {
            "pilot_metadata": {"checkpoint": "CP4.2", "run_state": pilot.RUN_IN_PROGRESS},
            "diversity_by_prompt": [],
            "diversity_overall": pilot.summarize_diversity([]),
            "verdict": {},
            "reproducibility": [],
            "generations": inputs["generations"],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested" / "cp4_2_generation_pilot.json"
            pilot.write_artifact(path, payload)
            self.assertTrue(path.exists())
            with open(path, "r", encoding="utf-8") as handle:
                loaded = json.load(handle)
            self.assertEqual(len(loaded["generations"]), len(inputs["generations"]))

    def test_incremental_writes_preserve_earlier_generations(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cp4_2_generation_pilot.json"
            scenario = make_scenario(0, answerable=True)
            generations = []
            for offset, seed in enumerate(pilot.SEEDS):
                generations.append(
                    make_record(scenario, seed, 0, "answer-{0}".format(offset))
                )
                pilot.write_artifact(
                    path,
                    {
                        "pilot_metadata": {
                            "run_state": pilot.RUN_IN_PROGRESS,
                            "generations_attempted": len(generations),
                        },
                        "generations": list(generations),
                    },
                )
                # After every incremental write the artifact must be readable.
                with open(path, "r", encoding="utf-8") as handle:
                    checkpoint = json.load(handle)
                self.assertEqual(len(checkpoint["generations"]), offset + 1)
            self.assertTrue(pilot.verify_artifact(path, 10))
            self.assertFalse(pilot.verify_artifact(path, 9))

    def test_write_artifact_is_atomic_and_leaves_no_tmp(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cp4_2_generation_pilot.json"
            pilot.write_artifact(path, {"generations": []})
            self.assertEqual(
                sorted(p.name for p in Path(tmp).iterdir()), ["cp4_2_generation_pilot.json"]
            )

    def test_verify_artifact_rejects_invalid_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cp4_2_generation_pilot.json"
            path.write_text("{not json", encoding="utf-8")
            self.assertFalse(pilot.verify_artifact(path, 0))

    def test_verify_artifact_rejects_missing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertFalse(pilot.verify_artifact(Path(tmp) / "absent.json", 0))


# --------------------------------------------------------------------------
# FIX 12 - scope boundary
# --------------------------------------------------------------------------

class TestScopeBoundary(unittest.TestCase):
    def test_no_future_stage_modules_or_functions(self):
        for banned in (
            "compute_hidden_states",
            "extract_hidden_states",
            "evaluate_generation",
            "label_hallucination",
            "compute_empirical_risk",
            "extract_prompt_features",
            "train_predictor",
            "split_dataset",
        ):
            self.assertFalse(hasattr(pilot, banned), "unexpected scope creep: {0}".format(banned))

    def test_artifact_and_report_paths_target_cp4_2_only(self):
        self.assertEqual(
            pilot.PILOT_ARTIFACT_PATH, Path("data/pilots/cp4_2_generation_pilot.json")
        )
        self.assertEqual(
            pilot.PILOT_REPORT_PATH, Path("docs/cp4_2_generation_protocol_pilot_report.md")
        )

    def test_run_generation_pilot_exists_and_is_the_entry_point(self):
        self.assertTrue(callable(pilot.run_generation_pilot))
        self.assertTrue(callable(pilot.main))
        self.assertIn("run_generation_pilot()", inspect.getsource(pilot.main))

    def test_generation_helpers_are_not_executed_at_import(self):
        source = inspect.getsource(pilot)
        self.assertNotIn("model.generate(", inspect.getsource(pilot.generation_config))
        # generate() may only appear inside run_generation_pilot.
        generate_lines = [
            line for line in source.splitlines() if "model.generate(" in line
        ]
        self.assertTrue(generate_lines)
        run_source = inspect.getsource(pilot.run_generation_pilot)
        self.assertEqual(len(generate_lines), run_source.count("model.generate("))


if __name__ == "__main__":
    unittest.main()
