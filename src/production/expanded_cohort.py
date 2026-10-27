"""Selection and preflight helpers for the expanded SQuAD v2 cohort."""

import hashlib
import json
import os
import random
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple


TARGET_PROMPTS = 500
BATCH_SIZE = 100
BATCH_COUNT = TARGET_PROMPTS // BATCH_SIZE
SELECTION_SEED = 20261017
GENERATIONS_PER_PROMPT = 10
SEEDS = tuple(range(1001, 1011))
EVALUATOR_MODEL = "gemini-3.5-flash-lite"
PROMPT_VERSION = "CP4.7.0"
MODEL_NAME = "Qwen/Qwen3-1.7B"
MODEL_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
SELECTION_METHOD = "Preserve existing prompts; seeded random sample without replacement from remaining unique canonical question/context pairs."


def batch_scenarios(selection: Dict[str, Any], batch_id: int) -> List[Dict[str, Any]]:
    """Return the stable, one-based 100-prompt batch from a saved selection."""
    rows = selection.get("selected_scenarios")
    if not isinstance(rows, list) or len(rows) != selection.get("target_prompts"):
        raise ValueError("Saved selection count does not match target_prompts")
    if batch_id < 1 or batch_id > BATCH_COUNT:
        raise ValueError(f"batch_id must be between 1 and {BATCH_COUNT}")
    if selection.get("target_prompts") == TARGET_PROMPTS:
        if len(rows) != TARGET_PROMPTS or selection.get("seeds") != list(SEEDS):
            raise ValueError("Saved selection does not match the frozen 500-prompt protocol")
        prompt_ids = [row.get("prompt_id") for row in rows]
        content_hashes = [row.get("content_sha256") for row in rows]
        if len(set(prompt_ids)) != TARGET_PROMPTS:
            raise ValueError("Saved selection contains duplicate prompt IDs")
        if len(set(content_hashes)) != TARGET_PROMPTS:
            raise ValueError("Saved selection contains duplicate prompt content")
    elif batch_id != 1 or len(rows) > BATCH_SIZE:
        raise ValueError("Only the fixed 500-prompt cohort supports five 100-prompt batches")
    start = (batch_id - 1) * BATCH_SIZE
    result = rows[start : start + BATCH_SIZE]
    if not result:
        raise ValueError(f"Batch {batch_id} contains no selected prompts")
    return result


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def atomic_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def source_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selection_digest(selection: Dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(selection, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _normalized_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def scenario_content_key(scenario: Dict[str, Any]) -> Tuple[str, str]:
    prompt = scenario.get("prompt", {})
    question = prompt.get("question")
    context = prompt.get("context")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Canonical scenario is missing a question")
    if not isinstance(context, str) or not context.strip():
        raise ValueError("Canonical scenario is missing a context")
    return _normalized_text(question), _normalized_text(context)


def select_expanded_scenarios(
    scenarios: List[Dict[str, Any]],
    existing_manifest: Dict[str, Any],
    target_prompts: int = TARGET_PROMPTS,
    seed: int = SELECTION_SEED,
) -> List[Dict[str, Any]]:
    """Return a deterministic unique cohort that retains the existing prompts."""
    if target_prompts <= 0:
        raise ValueError("target_prompts must be positive")

    by_id: Dict[str, Tuple[int, Dict[str, Any]]] = {}
    first_by_content: Dict[Tuple[str, str], Tuple[int, Dict[str, Any]]] = {}
    duplicate_count = 0
    for index, scenario in enumerate(scenarios):
        source = scenario.get("source", {})
        if source.get("dataset") != "squad_v2" or source.get("split") != "train":
            continue
        sample_id = scenario.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"Canonical scenario at index {index} has no sample_id")
        if sample_id in by_id:
            raise ValueError(f"Duplicate canonical sample_id: {sample_id}")
        by_id[sample_id] = (index, scenario)
        content = scenario_content_key(scenario)
        if content in first_by_content:
            duplicate_count += 1
            continue
        first_by_content[content] = (index, scenario)

    existing_by_id: Dict[str, int] = {}
    for generation in existing_manifest.get("generations", []):
        prompt_id = generation.get("prompt_id")
        dataset_index = generation.get("dataset_index")
        if prompt_id not in by_id:
            raise ValueError(f"Existing prompt {prompt_id!r} is absent from canonical data")
        canonical_index, _ = by_id[prompt_id]
        if canonical_index != dataset_index:
            raise ValueError(
                f"Existing prompt {prompt_id!r} changed canonical dataset index"
            )
        existing_by_id[prompt_id] = dataset_index

    existing_content: Dict[Tuple[str, str], str] = {}
    for prompt_id, index in existing_by_id.items():
        scenario = scenarios[index]
        content = scenario_content_key(scenario)
        previous = existing_content.get(content)
        if previous is not None and previous != prompt_id:
            raise ValueError(
                f"Existing cohort contains duplicate prompt content: {previous}, {prompt_id}"
            )
        existing_content[content] = prompt_id

    available_unique = len(first_by_content)
    if target_prompts > available_unique:
        raise ValueError(
            f"Requested {target_prompts} unique prompts, but only "
            f"{available_unique} are available"
        )
    if len(existing_by_id) > target_prompts:
        raise ValueError(
            f"Existing cohort has {len(existing_by_id)} prompts, exceeding target "
            f"{target_prompts}"
        )

    selected: List[Tuple[int, Dict[str, Any], bool]] = []
    for prompt_id, index in sorted(existing_by_id.items(), key=lambda item: item[1]):
        selected.append((index, scenarios[index], True))

    candidates = [
        (index, scenario)
        for content, (index, scenario) in first_by_content.items()
        if content not in existing_content
    ]
    rng = random.Random(seed)
    additions = rng.sample(candidates, target_prompts - len(selected))
    selected.extend((index, scenario, False) for index, scenario in additions)

    result = []
    content_seen = set()
    ids_seen = set()
    for index, scenario, existing in selected:
        prompt_id = scenario["sample_id"]
        content = scenario_content_key(scenario)
        if prompt_id in ids_seen or content in content_seen:
            raise AssertionError("Selected cohort contains a duplicate prompt")
        ids_seen.add(prompt_id)
        content_seen.add(content)
        result.append({
            "dataset_index": index,
            "prompt_id": prompt_id,
            "source_dataset": scenario["source"]["dataset"],
            "source_split": scenario["source"]["split"],
            "content_sha256": hashlib.sha256(
                "\0".join(content).encode("utf-8")
            ).hexdigest(),
            "preserved_existing_prompt": existing,
        })
    return result


def create_selection_artifact(
    scenarios_path: Path,
    source_generations_path: Path,
    target_prompts: int = TARGET_PROMPTS,
    seed: int = SELECTION_SEED,
) -> Dict[str, Any]:
    scenarios = load_json(scenarios_path)
    existing_manifest = load_json(source_generations_path)
    selected = select_expanded_scenarios(
        scenarios, existing_manifest, target_prompts, seed
    )
    canonical_squad_train = [
        scenario for scenario in scenarios
        if scenario.get("source", {}).get("dataset") == "squad_v2"
        and scenario.get("source", {}).get("split") == "train"
    ]
    canonical_unique = {
        scenario_content_key(scenario) for scenario in canonical_squad_train
    }
    return {
        "schema_version": "LUX-COHORT-SELECTION-1",
        "target_prompts": target_prompts,
        "generations_per_prompt": GENERATIONS_PER_PROMPT,
        "seeds": list(SEEDS),
        "selection_seed": seed,
        "selection_method": SELECTION_METHOD,
        "duplicate_rule": (
            "Question and context are Unicode NFKC normalized, case-folded, "
            "and whitespace-collapsed before comparing."
        ),
        "source_dataset": "squad_v2",
        "source_split": "train",
        "canonical_scenario_count": len(canonical_squad_train),
        "canonical_unique_prompt_count": len(canonical_unique),
        "canonical_duplicate_content_count": (
            len(canonical_squad_train) - len(canonical_unique)
        ),
        "canonical_scenarios_path": str(scenarios_path),
        "canonical_scenarios_sha256": source_hash(scenarios_path),
        "existing_generations_path": str(source_generations_path),
        "existing_generations_sha256": source_hash(source_generations_path),
        "existing_source_prompts": len({
            generation["prompt_id"]
            for generation in existing_manifest.get("generations", [])
        }),
        "unique_prompts_selected": len(selected),
        "selected_scenarios": selected,
    }


def generation_matches_protocol(generation: Dict[str, Any]) -> bool:
    expected_id = "{0}:{1}:{2}".format(
        generation.get("prompt_id"),
        generation.get("dataset_index"),
        generation.get("seed"),
    )
    return (
        generation.get("prompt_id") == generation.get("sample_id")
        and generation.get("generation_id") == expected_id
        and generation.get("generation_index")
        == generation.get("seed", -1) - SEEDS[0]
        and generation.get("model_name") == MODEL_NAME
        and generation.get("model_revision") == MODEL_REVISION
        and generation.get("device") == "cpu"
        and generation.get("thinking_mode") is False
        and generation.get("do_sample") is True
        and generation.get("temperature") == 0.7
        and generation.get("top_p") == 0.8
        and generation.get("top_k") == 20
        and generation.get("max_new_tokens") == 512
        and generation.get("seed") in SEEDS
    )


def _compatible_generation(generation: Dict[str, Any]) -> bool:
    return (
        generation.get("generation_status") == "SUCCESS"
        and generation_matches_protocol(generation)
    )


def _read_latest_evaluations(
    production_dir: Path, selected_generation_ids: set
) -> Dict[str, Dict[str, Any]]:
    latest: Dict[str, Dict[str, Any]] = {}
    for evaluations_dir in production_dir.rglob("evaluations"):
        if not evaluations_dir.is_dir():
            continue
        for path in evaluations_dir.rglob("*.json"):
            try:
                record = load_json(path)
            except (OSError, ValueError):
                continue
            generation_id = record.get("source_generation_id")
            if generation_id not in selected_generation_ids:
                continue
            key = (record.get("evaluator_model"), record.get("prompt_version"), generation_id)
            timestamp = record.get("timestamp", "")
            previous = latest.get("\0".join(key))
            if previous is None or timestamp >= previous.get("timestamp", ""):
                latest["\0".join(key)] = record
    return latest


def build_preflight_report(
    selection: Dict[str, Any],
    source_manifest: Dict[str, Any],
    production_dir: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    selected = selection["selected_scenarios"]
    selected_ids = {row["prompt_id"] for row in selected}
    generations_by_prompt: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for generation in source_manifest.get("generations", []):
        if generation.get("prompt_id") in selected_ids:
            generations_by_prompt[generation["prompt_id"]].append(generation)

    compatible_prompts = set()
    compatible_generations = 0
    incompatible_generations = 0
    existing_generation_ids = set()
    reusable_generation_ids = set()
    for prompt_id in selected_ids:
        records = generations_by_prompt.get(prompt_id, [])
        seeds = [record.get("seed") for record in records]
        valid_records = {
            record["generation_id"]: record
            for record in records
            if record.get("generation_id") and _compatible_generation(record)
        }
        compatible_generations += len(valid_records)
        reusable_generation_ids.update(
            valid_records
        )
        all_compatible = (
            len(records) == GENERATIONS_PER_PROMPT
            and set(seeds) == set(SEEDS)
            and len(seeds) == len(set(seeds))
            and len(valid_records) == GENERATIONS_PER_PROMPT
        )
        if all_compatible:
            compatible_prompts.add(prompt_id)
        incompatible_generations += sum(
            1 for record in records if not generation_matches_protocol(record)
        )
        existing_generation_ids.update(
            record.get("generation_id")
            for record in records
            if record.get("generation_id")
        )

    expected_generations = len(selected) * GENERATIONS_PER_PROMPT
    missing_generations = expected_generations - compatible_generations
    global_hidden_dir = production_dir / "hidden_states"
    expanded_hidden_dir = output_dir / "hidden_states"
    existing_hidden_prompts = 0
    for row in selected:
        index = row["dataset_index"]
        candidates = [
            global_hidden_dir / f"hs_{index}.json",
            expanded_hidden_dir / f"hs_{index}.json",
        ]
        compatible_hidden = False
        for path in candidates:
            if not path.is_file():
                continue
            try:
                hidden = load_json(path)
            except (OSError, ValueError):
                continue
            if (
                hidden.get("prompt_dataset_index") == index
                and hidden.get("model_name") == MODEL_NAME
                and hidden.get("model_revision") == MODEL_REVISION
            ):
                compatible_hidden = True
                break
        if compatible_hidden:
            existing_hidden_prompts += 1

    latest_evaluations = _read_latest_evaluations(
        production_dir, existing_generation_ids
    )
    eval_counts: Dict[str, Counter] = defaultdict(Counter)
    cp47_success_ids = set()
    for record in latest_evaluations.values():
        version_key = f"{record.get('evaluator_model')} / {record.get('prompt_version')}"
        eval_counts[version_key][record.get("evaluation_status", "UNKNOWN")] += 1
        if (
            record.get("evaluator_model") == EVALUATOR_MODEL
            and record.get("prompt_version") == PROMPT_VERSION
            and record.get("evaluation_status") == "SUCCESS"
            and record.get("source_generation_id") in reusable_generation_ids
        ):
            cp47_success_ids.add(record["source_generation_id"])

    new_generation_prompts = len(selected) - len(compatible_prompts)
    known_evaluation_calls = len(cp47_success_ids)
    expected_evaluation_calls = (
        expected_generations - known_evaluation_calls
    )

    return {
        "report_version": "LUX-EXPANDED-PREFLIGHT-1",
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "source_dataset": selection["source_dataset"],
        "source_split": selection["source_split"],
        "canonical_scenario_count": selection.get("canonical_scenario_count"),
        "canonical_unique_prompt_count": selection.get("canonical_unique_prompt_count"),
        "canonical_duplicate_content_count": selection.get(
            "canonical_duplicate_content_count"
        ),
        "canonical_scenarios_sha256": selection["canonical_scenarios_sha256"],
        "selection": {
            "target_prompt_count": len(selected),
            "selection_seed": selection["selection_seed"],
            "selection_method": selection["selection_method"],
            "duplicate_rule": selection["duplicate_rule"],
            "selection_artifact": str(output_dir / "selection.json"),
            "selected_prompt_ids_and_indices_recorded": True,
            "preserved_existing_prompt_count": selection["existing_source_prompts"],
            "new_prompt_count": new_generation_prompts,
        },
        "generation": {
            "generations_per_prompt": GENERATIONS_PER_PROMPT,
            "expected_total_generations": expected_generations,
            "compatible_existing_prompts": len(compatible_prompts),
            "compatible_existing_generations": compatible_generations,
            "incompatible_existing_generation_records": incompatible_generations,
            "new_generation_prompts_required": new_generation_prompts,
            "missing_generation_work": missing_generations,
            "expected_generation_settings": {
                "model_name": MODEL_NAME,
                "model_revision": MODEL_REVISION,
                "device": "cpu",
                "thinking_mode": False,
                "do_sample": True,
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 20,
                "max_new_tokens": 512,
                "seeds": list(SEEDS),
            },
        },
        "hidden_states": {
            "selected_prompts_with_existing_artifact": existing_hidden_prompts,
            "selected_prompts_missing_artifact": len(selected) - existing_hidden_prompts,
            "existing_artifacts_will_be_copied_not_modified": True,
        },
        "evaluations": {
            "fixed_evaluator_model": EVALUATOR_MODEL,
            "fixed_prompt_version": PROMPT_VERSION,
            "existing_counts_by_evaluator_version_and_status": {
                version: dict(counts) for version, counts in sorted(eval_counts.items())
            },
            "existing_cp4_7_successful_generations": known_evaluation_calls,
            "additional_evaluator_calls_expected": expected_evaluation_calls,
            "target_eligibility": (
                "A prompt is eligible only when exactly ten matching-version "
                "records are SUCCESS with RELIABLE or UNRELIABLE labels. "
                "INADEQUATE, FAILED, MANUAL_REVIEW, missing, and incomplete "
                "records are excluded; risk is UNRELIABLE / 10."
            ),
            "evaluator_versions_are_never_mixed": True,
            "live_quota_checked": False,
            "quota_risk_note": (
                "No quota probe was made. Historical CP4.7.0 records include a "
                "429 rate-limit failure; current quota and runtime are unknown."
            ),
        },
        "predictor": {
            "target": "continuous prompt-level empirical risk",
            "inputs": [
                "pre-generation prompt features",
                "pre-generation frozen-model hidden-state features",
            ],
            "forbidden_inputs": [
                "generated answers",
                "evaluator labels or status",
                "claim evaluations",
                "empirical risk",
                "post-generation latency",
            ],
            "split": "prompt identity; shared across baseline and all three models",
            "preprocessing": "fit on training prompts only",
        },
        "runtime_and_cost": {
            "estimate": None,
            "reason": "No measured expanded-cohort runtime or provider pricing is available.",
        },
        "staged_commands": {
            "preflight": "python run_expanded_cohort.py preflight --target-prompts 500 --seed 20261017",
            "generation": "python run_expanded_cohort.py generate --max-new-prompts 10",
            "evaluation": (
                "python run_eval_cp4_7.py evaluate --generation-path "
                "data/production/squad_v2_expanded_500/generations.json "
                "--max-evaluations 100 --spacing 5"
            ),
            "targets": (
                "python run_eval_cp4_7.py build-targets --generation-path "
                "data/production/squad_v2_expanded_500/generations.json "
                "--target-path data/production/squad_v2_expanded_500/targets.json"
            ),
            "predictor_features": "python run_expanded_cohort.py build-features",
            "predictors": (
                "python run_predictor.py compare --features-path "
                "data/production/squad_v2_expanded_500/predictor_features.json "
                "--targets-path data/production/squad_v2_expanded_500/targets.json "
                "--hidden-states-dir data/production/squad_v2_expanded_500/hidden_states "
                "--output-dir data/production/squad_v2_expanded_500/predictors"
            ),
        },
    }


def create_predictor_feature_artifact(
    selection: Dict[str, Any],
    canonical_scenarios: List[Dict[str, Any]],
    hidden_states_dir: Path,
    token_counter=None,
) -> Dict[str, Any]:
    """Build feature rows from canonical pre-generation data only."""
    rows = []
    for selected in selection["selected_scenarios"]:
        index = selected["dataset_index"]
        scenario = canonical_scenarios[index]
        prompt = scenario["prompt"]
        question = prompt["question"]
        context = prompt["context"]
        hidden_path = hidden_states_dir / f"hs_{index}.json"
        if not hidden_path.is_file():
            raise FileNotFoundError(f"Missing hidden state artifact: {hidden_path}")
        hidden = load_json(hidden_path)
        if (
            hidden.get("prompt_dataset_index") != index
            or hidden.get("model_name") != MODEL_NAME
            or hidden.get("model_revision") != MODEL_REVISION
        ):
            raise ValueError(f"Incompatible hidden state artifact: {hidden_path}")
        metrics = {
            "question_chars": len(question),
            "context_chars": len(context),
            "question_ends_with_question_mark": question.rstrip().endswith("?"),
        }
        if token_counter is not None:
            metrics["question_tokens"] = token_counter(question)
            metrics["context_tokens"] = token_counter(context)
        rows.append({
            "prompt_id": selected["prompt_id"],
            "dataset_index": index,
            "hidden_state_ref": str(hidden_path),
            "prompt_metrics": metrics,
        })
    return {
        "pipeline": {
            "artifact": "expanded_500_prompt_predictor_features",
            "model_revision": MODEL_REVISION,
            "feature_timing": "pre-generation only",
            "excluded_fields": [
                "generated_answer",
                "generation_label",
                "claim_evaluations",
                "evaluation_status",
                "empirical_hallucination_risk",
                "status",
                "latency",
                "generated_token_count",
                "answerability",
            ],
        },
        "predictor_features": rows,
    }
