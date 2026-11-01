#!/usr/bin/env python
"""Prepare, preflight, and incrementally generate the expanded 500-prompt cohort.

This command does not evaluate generations. Evaluator requests are a separate,
explicit CP4.7.0 stage. The ``preflight`` command is the safe first action.
"""

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.production.expanded_cohort import (
    BATCH_COUNT,
    BATCH_SIZE,
    EVALUATOR_MODEL,
    MODEL_NAME,
    MODEL_REVISION,
    PROMPT_VERSION,
    SEEDS,
    SELECTION_SEED,
    TARGET_PROMPTS,
    atomic_json,
    batch_scenarios,
    build_preflight_report,
    create_selection_artifact,
    create_predictor_feature_artifact,
    generation_matches_protocol,
    load_json,
    source_hash,
    selection_digest,
    select_expanded_scenarios,
)

CANONICAL_PATH = ROOT / "data" / "prompts" / "squad_v2_canonical_scenarios.json"
SOURCE_GENERATIONS_PATH = ROOT / "data" / "production" / "generations.json"
OUTPUT_DIR = ROOT / "data" / "production" / "squad_v2_expanded_500"


def _selection_for_run(
    output_dir: Path, target_prompts: int, seed: int
) -> Dict[str, Any]:
    selection_path = output_dir / "selection.json"
    if selection_path.exists():
        selection = load_json(selection_path)
        if (
            selection.get("target_prompts") != target_prompts
            or selection.get("selection_seed") != seed
            or selection.get("canonical_scenarios_sha256") != source_hash(CANONICAL_PATH)
            or selection.get("existing_generations_sha256")
            != source_hash(SOURCE_GENERATIONS_PATH)
        ):
            raise ValueError(
                f"{selection_path} belongs to a different cohort/source; "
                "use a new --output-dir rather than replacing it"
            )
        canonical = load_json(CANONICAL_PATH)
        existing = load_json(SOURCE_GENERATIONS_PATH)
        recomputed = select_expanded_scenarios(
            canonical, existing, target_prompts, seed
        )
        if recomputed != selection.get("selected_scenarios"):
            raise ValueError("Saved selection does not match deterministic selection")
        return selection

    selection = create_selection_artifact(
        CANONICAL_PATH, SOURCE_GENERATIONS_PATH, target_prompts, seed
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(selection_path, selection)
    return selection


def run_preflight(output_dir: Path, target_prompts: int, seed: int) -> Dict[str, Any]:
    selection = _selection_for_run(output_dir, target_prompts, seed)
    source_manifest = load_json(SOURCE_GENERATIONS_PATH)
    report = build_preflight_report(
        selection, source_manifest, ROOT / "data" / "production", output_dir
    )
    report["canonical_scenario_count"] = selection["canonical_scenario_count"]
    report["canonical_unique_prompt_count"] = selection[
        "canonical_unique_prompt_count"
    ]
    atomic_json(output_dir / "preflight.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def _safe_generation_name(generation_id: str) -> str:
    return generation_id.replace(":", "_").replace("/", "_") + ".json"


def _latest_generation_records(manifest: Dict[str, Any], attempts_dir: Path):
    latest = {
        generation["generation_id"]: generation
        for generation in manifest.get("generations", [])
    }
    attempt_paths = sorted(attempts_dir.glob("*.json"))
    retry_dir = attempts_dir / "retries"
    if retry_dir.exists():
        attempt_paths.extend(sorted(retry_dir.glob("*.json")))
    for path in attempt_paths:
        if path.exists():
            record = load_json(path)
            latest[record["generation_id"]] = record
    return latest


def _sync_generation_manifest(
    manifest: Dict[str, Any],
    selection: Dict[str, Any],
    latest_records: Dict[str, Dict[str, Any]],
) -> None:
    records = []
    for selected in selection["selected_scenarios"]:
        index = selected["dataset_index"]
        prompt_id = selected["prompt_id"]
        for seed in SEEDS:
            generation_id = f"{prompt_id}:{index}:{seed}"
            record = latest_records.get(generation_id)
            if record is not None:
                records.append(record)
    manifest["generations"] = records
    manifest["scenarios"] = [
        {"dataset_index": row["dataset_index"], "sample_id": row["prompt_id"]}
        for row in selection["selected_scenarios"]
    ]
    manifest["cohort_selection"] = {
        "selection_seed": selection["selection_seed"],
        "selection_method": selection["selection_method"],
        "selection_artifact_sha256": selection_digest(selection),
        "target_prompt_count": selection["target_prompts"],
        "source_dataset": selection["source_dataset"],
        "source_split": selection["source_split"],
        "source_generations_sha256": selection["existing_generations_sha256"],
    }
    manifest.setdefault("pipeline", {})["num_prompts"] = len(
        selection["selected_scenarios"]
    )
    manifest["pipeline"]["generations_per_prompt"] = len(SEEDS)
    manifest["pipeline"]["seeds"] = list(SEEDS)
    manifest.setdefault("summary", {})
    manifest["summary"].update({
        "n_prompts": len(selection["selected_scenarios"]),
        "n_generations": len(records),
        "n_success": sum(
            record.get("generation_status") == "SUCCESS" for record in records
        ),
        "n_failed": sum(
            record.get("generation_status") != "SUCCESS" for record in records
        ),
        "n_with_hidden_states": len({
            record["dataset_index"] for record in records
            if record.get("hidden_state_reference")
        }),
    })


def _validate_source_protocol(source_manifest: Dict[str, Any]) -> None:
    model = source_manifest.get("model", {})
    if model.get("name") != MODEL_NAME or model.get("revision") != MODEL_REVISION:
        raise ValueError("Existing generations do not match the frozen target model")
    if (
        model.get("device") != "cpu"
        or model.get("thinking_mode") is not False
        or model.get("decoding")
        != {
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "max_new_tokens": 512,
        }
    ):
        raise ValueError("Existing generation settings differ from the frozen protocol")
    if tuple(source_manifest.get("pipeline", {}).get("seeds", [])) != SEEDS:
        raise ValueError("Existing generation seed protocol is not 1001-1010")


def _initialise_generation_artifact(
    output_dir: Path, selection: Dict[str, Any], source_manifest: Dict[str, Any]
) -> Dict[str, Any]:
    generations_path = output_dir / "generations.json"
    selection_sha = selection_digest(selection)
    if generations_path.exists():
        manifest = load_json(generations_path)
        stored_digest = manifest.get("cohort_selection", {}).get(
            "selection_artifact_sha256"
        )
        if stored_digest != selection_sha:
            raise ValueError("Expanded generation manifest belongs to another selection")
        return manifest

    _validate_source_protocol(source_manifest)
    existing_prompt_ids = {
        row["prompt_id"] for row in selection["selected_scenarios"]
        if row["preserved_existing_prompt"]
    }
    copied_generations = [
        generation for generation in source_manifest.get("generations", [])
        if generation.get("prompt_id") in existing_prompt_ids
    ]
    if any(not generation_matches_protocol(record) for record in copied_generations):
        raise ValueError(
            "An existing generation does not match the cohort's frozen model, "
            "seed, settings, or stable generation ID"
        )
    manifest = json.loads(json.dumps(source_manifest))
    manifest["pipeline"] = dict(manifest.get("pipeline", {}))
    manifest["pipeline"].update({
        "checkpoint": "LUX expanded 500-prompt SQuAD v2 cohort",
        "phase": "generation_in_progress",
    })
    manifest["generations"] = copied_generations
    manifest["summary"] = {}
    _sync_generation_manifest(
        manifest,
        selection,
        {record["generation_id"]: record for record in copied_generations},
    )
    atomic_json(generations_path, manifest)
    return manifest


def _hidden_state_is_compatible(path: Path, dataset_index: int) -> bool:
    if not path.is_file():
        return False
    value = load_json(path)
    return (
        value.get("prompt_dataset_index") == dataset_index
        and value.get("model_name") == MODEL_NAME
        and value.get("model_revision") == MODEL_REVISION
    )


def _ensure_preserved_hidden_states(
    output_dir: Path, selection: Dict[str, Any]
) -> None:
    source_hidden_dir = ROOT / "data" / "production" / "hidden_states"
    destination_hidden_dir = output_dir / "hidden_states"
    for row in selection["selected_scenarios"]:
        if not row["preserved_existing_prompt"]:
            continue
        index = row["dataset_index"]
        source = source_hidden_dir / f"hs_{index}.json"
        destination = destination_hidden_dir / source.name
        if not _hidden_state_is_compatible(source, index):
            raise FileNotFoundError(
                f"Compatible existing hidden state is missing or invalid: {source}"
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if source_hash(source) != source_hash(destination):
                raise ValueError(f"Existing expanded hidden state differs: {destination}")
        else:
            temporary = destination.with_name(destination.name + ".tmp")
            temporary.write_bytes(source.read_bytes())
            temporary.replace(destination)


def _batch_record_path(output_dir: Path, batch_id: int) -> Path:
    return output_dir / "batches" / f"batch_{batch_id:02d}.json"


def _batch_generation_ids(rows: List[Dict[str, Any]]) -> List[str]:
    return [
        f"{row['prompt_id']}:{row['dataset_index']}:{seed}"
        for row in rows
        for seed in SEEDS
    ]


def _batch_record(
    output_dir: Path,
    selection: Dict[str, Any],
    batch_id: int,
    rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    path = _batch_record_path(output_dir, batch_id)
    expected = {
        "schema_version": "LUX-COHORT-BATCH-1",
        "batch_id": f"batch_{batch_id:02d}",
        "batch_number": batch_id,
        "batch_size": BATCH_SIZE,
        "selection_artifact_sha256": selection_digest(selection),
        "selected_prompt_ids": [row["prompt_id"] for row in rows],
        "expected_generation_ids": _batch_generation_ids(rows),
        "generation_configuration": {
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
        "evaluator_configuration": {
            "model": EVALUATOR_MODEL,
            "prompt_version": PROMPT_VERSION,
        },
    }
    if path.exists():
        record = load_json(path)
        if any(record.get(key) != value for key, value in expected.items()):
            raise ValueError(
            f"Batch record does not match the saved cohort/protocol: {path}"
            )
        return record
    from run_production import _now_iso

    record = {
        **expected,
        "status": "pending",
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
        "started_at": None,
        "completed_at": None,
        "attempted_generation_ids": [],
        "successful_generation_ids": [],
        "failed_generation_ids": [],
        "evaluation_status": "pending",
    }
    atomic_json(path, record)
    return record


def run_batch_preflight(
    output_dir: Path,
    batch_id: int,
    credential_slots: Optional[List[str]] = None,
    genai_sdk_available: bool = False,
) -> Dict[str, Any]:
    selection = _selection_for_run(output_dir, TARGET_PROMPTS, SELECTION_SEED)
    rows = batch_scenarios(selection, batch_id)
    if len(rows) != BATCH_SIZE or selection.get("target_prompts") != TARGET_PROMPTS:
        raise ValueError("Batch execution requires the saved 500-prompt selection")
    source_manifest = load_json(SOURCE_GENERATIONS_PATH)
    if source_hash(SOURCE_GENERATIONS_PATH) != selection["existing_generations_sha256"]:
        raise ValueError("Original generation artifact changed after cohort selection")
    _validate_source_protocol(source_manifest)
    if EVALUATOR_MODEL != "gemini-3.5-flash-lite" or PROMPT_VERSION != "CP4.7.0":
        raise ValueError("Configured evaluator differs from the fixed CP4.7.0 protocol")

    expanded_path = output_dir / "generations.json"
    expanded_manifest = load_json(expanded_path) if expanded_path.exists() else None
    manifest = expanded_manifest or source_manifest
    if expanded_manifest:
        if expanded_manifest.get("cohort_selection", {}).get(
            "selection_artifact_sha256"
        ) != selection_digest(selection):
            raise ValueError("Expanded generation manifest belongs to another selection")
    attempts = _latest_generation_records(
        manifest, output_dir / "generation_attempts"
    )
    for generation in source_manifest.get("generations", []):
        attempts.setdefault(generation["generation_id"], generation)

    expected_ids = set(_batch_generation_ids(rows))
    batch_records = {
        generation_id: attempts[generation_id]
        for generation_id in expected_ids
        if generation_id in attempts
    }
    compatible = {
        generation_id: record
        for generation_id, record in batch_records.items()
        if record.get("generation_status") == "SUCCESS"
        and generation_matches_protocol(record)
    }
    incompatible = sum(
        not generation_matches_protocol(record)
        for record in batch_records.values()
    )

    hidden_existing = 0
    for row in rows:
        index = row["dataset_index"]
        sources = (
            output_dir / "hidden_states" / f"hs_{index}.json",
            ROOT / "data" / "production" / "hidden_states" / f"hs_{index}.json",
        )
        if any(_hidden_state_is_compatible(path, index) for path in sources):
            hidden_existing += 1

    evaluation_statuses = defaultdict(int)
    eval_dir = ROOT / "data" / "production" / "cp4_7_0_gemini_3_5_flash_lite" / "evaluations"
    resolved_evaluation_ids = set()
    for generation_id in expected_ids:
        safe_name = _safe_generation_name(generation_id)
        paths = [eval_dir / safe_name]
        retry_dir = eval_dir / "retries"
        if retry_dir.exists():
            paths.extend(sorted(retry_dir.glob(Path(safe_name).stem + ".retry_*.json")))
        records = []
        for path in paths:
            if not path.exists():
                continue
            record = load_json(path)
            if (
                record.get("source_generation_id") != generation_id
                or record.get("evaluator_model") != EVALUATOR_MODEL
                or record.get("prompt_version") != PROMPT_VERSION
            ):
                raise ValueError(
                    f"Evaluation record has mismatched provenance: {path}"
                )
            records.append(record)
        if records:
            latest_record = max(records, key=lambda record: record.get("timestamp", ""))
            status = latest_record.get("evaluation_status", "UNKNOWN")
            evaluation_statuses[status] += 1
            if status in ("SUCCESS", "MANUAL_REVIEW"):
                resolved_evaluation_ids.add(generation_id)

    calls_expected = len(expected_ids) - len(resolved_evaluation_ids)
    slots = sorted(set(credential_slots or []))
    evaluator_ready = bool(slots) and genai_sdk_available
    record = _batch_record(output_dir, selection, batch_id, rows)
    report = {
        "report_version": "LUX-BATCH-PREFLIGHT-1",
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "batch_id": record["batch_id"],
        "batch_number": batch_id,
        "batch_size": BATCH_SIZE,
        "batch_count": BATCH_COUNT,
        "selection_artifact_sha256": selection_digest(selection),
        "selected_prompt_ids": [row["prompt_id"] for row in rows],
        "generation": {
            "expected_generations": len(expected_ids),
            "existing_compatible_generations": len(compatible),
            "new_generations_required": len(expected_ids) - len(compatible),
            "existing_incompatible_generation_records": incompatible,
            "generation_protocol": record["generation_configuration"],
        },
        "hidden_states": {
            "existing_compatible_prompt_artifacts": hidden_existing,
            "missing_prompt_artifacts": BATCH_SIZE - hidden_existing,
        },
        "evaluation": {
            "model": EVALUATOR_MODEL,
            "prompt_version": PROMPT_VERSION,
            "existing_record_status_counts": dict(evaluation_statuses),
            "additional_calls_expected": calls_expected,
            "configured_credential_slots": slots,
            "configured_credential_count": len(slots),
            "google_genai_sdk_available": genai_sdk_available,
            "historical_success_proves_prior_model_access": any(
            record.get("evaluator_model") == EVALUATOR_MODEL
            and record.get("prompt_version") == PROMPT_VERSION
            and record.get("evaluation_status") == "SUCCESS"
            for path in eval_dir.glob("*.json")
            for record in [load_json(path)]
            ),
            "current_quota": "unknown; no quota probe performed",
            "historical_rate_limit_evidence": (
            "A saved CP4.7.0 attempt reported HTTP 429 with a "
            "15 requests/minute/project/model free-tier limit."
            ),
            "configured_spacing_seconds": 5,
            "configured_max_seconds": 9000,
            "configured_quota_cooldown_seconds": 60,
        },
        "can_execute": evaluator_ready and incompatible == 0,
        "blockers": (
            []
            if evaluator_ready and incompatible == 0
            else [
            reason
            for reason, condition in (
                ("No configured Gemini credentials", not slots),
                ("Google GenAI SDK is unavailable", not genai_sdk_available),
                ("Incompatible existing generation records", incompatible > 0),
            )
            if condition
            ]
        ),
        "batch_record_path": str(_batch_record_path(output_dir, batch_id)),
    }
    report_path = output_dir / "batches" / f"batch_{batch_id:02d}_preflight.json"
    atomic_json(report_path, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def _write_generation_attempt(
    attempts_dir: Path, record: Dict[str, Any]
) -> Path:
    base_path = attempts_dir / _safe_generation_name(record["generation_id"])
    existing_paths = [base_path]
    retry_dir = attempts_dir / "retries"
    if retry_dir.exists():
        existing_paths.extend(sorted(retry_dir.glob(f"{base_path.stem}.retry_*.json")))
    existing_records = [load_json(path) for path in existing_paths if path.exists()]
    if any(item.get("generation_status") == "SUCCESS" for item in existing_records):
        raise ValueError(
            f"Refusing to write over a successful generation: {record['generation_id']}"
        )
    if not base_path.exists():
        path = base_path
    else:
        retry_dir.mkdir(parents=True, exist_ok=True)
        stem = base_path.stem
        retry_numbers = []
        for existing in retry_dir.glob(f"{stem}.retry_*.json"):
            try:
                retry_numbers.append(int(existing.stem.rsplit(".retry_", 1)[1]))
            except (IndexError, ValueError):
                continue
        path = retry_dir / f"{stem}.retry_{max(retry_numbers, default=0) + 1:03d}.json"
    atomic_json(path, record)
    return path


def run_generation(
    output_dir: Path,
    max_new_prompts: Optional[int],
    batch_id: int,
) -> Dict[str, Any]:
    if max_new_prompts is not None and (
        max_new_prompts <= 0 or max_new_prompts > BATCH_SIZE
    ):
        raise ValueError(f"--max-new-prompts must be between 1 and {BATCH_SIZE}")
    import run_production as production
    import run_cp4_2_pilot as cp42

    selection = load_json(output_dir / "selection.json")
    if source_hash(CANONICAL_PATH) != selection["canonical_scenarios_sha256"]:
        raise ValueError("Canonical source changed after cohort selection")
    source_manifest = load_json(SOURCE_GENERATIONS_PATH)
    if source_hash(SOURCE_GENERATIONS_PATH) != selection["existing_generations_sha256"]:
        raise ValueError("Original generation artifact changed after cohort selection")
    _validate_source_protocol(source_manifest)
    if select_expanded_scenarios(
        scenarios := load_json(CANONICAL_PATH),
        source_manifest,
        selection["target_prompts"],
        selection["selection_seed"],
    ) != selection["selected_scenarios"]:
        raise ValueError("Saved selection does not match deterministic selection")
    selected_batch = batch_scenarios(selection, batch_id)
    for previous_id in range(1, batch_id):
        previous_path = _batch_record_path(output_dir, previous_id)
        if not previous_path.exists() or load_json(previous_path).get("status") != "completed":
            raise ValueError(
                f"Batch {previous_id} must complete before generating batch {batch_id}"
            )
    batch_record = _batch_record(output_dir, selection, batch_id, selected_batch)
    expected_batch_ids = set(_batch_generation_ids(selected_batch))
    manifest = _initialise_generation_artifact(output_dir, selection, source_manifest)
    _ensure_preserved_hidden_states(output_dir, selection)
    scenarios_by_index = {i: scenario for i, scenario in enumerate(scenarios)}
    latest = _latest_generation_records(manifest, output_dir / "generation_attempts")
    _sync_generation_manifest(manifest, selection, latest)
    atomic_json(output_dir / "generations.json", manifest)

    incomplete = []
    for row in selected_batch:
        prompt_id = row["prompt_id"]
        index = row["dataset_index"]
        missing = [
            seed for seed in SEEDS
            if latest.get(f"{prompt_id}:{index}:{seed}", {}).get("generation_status")
            != "SUCCESS"
        ]
        if missing:
            incomplete.append((row, missing))
    if max_new_prompts is not None:
        incomplete = incomplete[:max_new_prompts]
    from run_production import _now_iso

    batch_record["status"] = "running"
    batch_record["started_at"] = batch_record.get("started_at") or _now_iso()
    batch_record["updated_at"] = _now_iso()
    atomic_json(_batch_record_path(output_dir, batch_id), batch_record)
    if not incomplete:
        batch_record["status"] = "completed"
        batch_record["completed_at"] = _now_iso()
        batch_record["attempted_generation_ids"] = sorted(
            generation_id for generation_id in expected_batch_ids if generation_id in latest
        )
        batch_record["successful_generation_ids"] = sorted(
            generation_id for generation_id in expected_batch_ids
            if latest.get(generation_id, {}).get("generation_status") == "SUCCESS"
        )
        batch_record["failed_generation_ids"] = []
        batch_record["updated_at"] = _now_iso()
        atomic_json(_batch_record_path(output_dir, batch_id), batch_record)
        return {
            "status": "complete",
            "batch_id": batch_record["batch_id"],
            "prompt_count": len(selected_batch),
            "generation_count": len(manifest["generations"]),
            "missing_generation_count": 0,
        }

    tokenizer, model = cp42._load_model()
    model.eval()
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    checkpoint_path = output_dir / "generation_checkpoint.json"
    checkpoint = load_json(checkpoint_path) if checkpoint_path.exists() else {
        "selection_seed": selection["selection_seed"],
        "started_at": production._now_iso(),
        "attempts": [],
    }
    checkpoint.setdefault("batches", {}).setdefault(
        batch_record["batch_id"], {"expected_generation_ids": sorted(expected_batch_ids)}
    )
    for row, missing_seeds in incomplete:
        index = row["dataset_index"]
        scenario = scenarios_by_index[index]
        if scenario.get("sample_id") != row["prompt_id"]:
            raise ValueError(f"Canonical scenario ID changed at dataset index {index}")
        hidden_path = output_dir / "hidden_states" / f"hs_{index}.json"
        if not _hidden_state_is_compatible(hidden_path, index):
            if hidden_path.exists():
                raise ValueError(
                    f"Refusing to overwrite an incompatible hidden state artifact: {hidden_path}"
                )
            if row["preserved_existing_prompt"]:
                raise FileNotFoundError(
                    f"Refusing to recapture preserved hidden state for prompt {row['prompt_id']}"
                )
            prompt_text = cp42.render_prompt(
                tokenizer,
                scenario["prompt"]["context"],
                scenario["prompt"]["question"],
            )
            hidden_payload = production.capture_hidden_states(
                model, tokenizer, prompt_text
            )
            production.atomic_write_json(hidden_path, {
                "prompt_dataset_index": index,
                "model_name": MODEL_NAME,
                "model_revision": MODEL_REVISION,
                "hidden_size": hidden_payload["hidden_size"],
                "num_layers": hidden_payload["num_layers"],
                "seq_len": hidden_payload["seq_len"],
                "last_token": hidden_payload["last_token"],
                "mean_pooled": hidden_payload["mean_pooled"],
            })
        else:
            prompt_text = cp42.render_prompt(
                tokenizer,
                scenario["prompt"]["context"],
                scenario["prompt"]["question"],
            )
        encoded = tokenizer(prompt_text, return_tensors="pt").to(production.DEVICE)
        prompt_token_count = int(encoded["input_ids"].shape[1])
        question = scenario["prompt"]["question"]
        context = scenario["prompt"]["context"]
        question_token_count = len(
            tokenizer(question, add_special_tokens=False)["input_ids"]
        )
        context_token_count = len(
            tokenizer(context, add_special_tokens=False)["input_ids"]
        )
        for seed in missing_seeds:
            generation_id = f"{row['prompt_id']}:{index}:{seed}"
            text, token_count, elapsed, error = production.generate_answer(
                model, tokenizer, encoded, seed, pad_token_id
            )
            status = "SUCCESS" if text and not error else "FAILED"
            record = production.build_production_record(
                scenario=scenario,
                dataset_index=index,
                generation_index=seed - SEEDS[0],
                seed=seed,
                prompt_text=prompt_text,
                prompt_token_count=prompt_token_count,
                status=status,
                generated_answer=text,
                generated_token_count=token_count,
                latency=elapsed,
                reached_max_tokens=token_count >= 512,
                error=error,
                hidden_state_ref=str(hidden_path.relative_to(ROOT)),
                question_token_count=question_token_count,
                context_token_count=context_token_count,
            )
            if record["generation_id"] != generation_id:
                raise ValueError("Generation ID construction diverged from the frozen protocol")
            attempt_path = _write_generation_attempt(
                output_dir / "generation_attempts", record
            )
            latest[generation_id] = record
            _sync_generation_manifest(manifest, selection, latest)
            atomic_json(output_dir / "generations.json", manifest)
            checkpoint.setdefault("attempts", []).append({
                "timestamp": production._now_iso(),
                "generation_id": generation_id,
                "status": status,
                "record_path": str(attempt_path.relative_to(ROOT)),
                "error": error,
            })
            checkpoint["last_updated"] = production._now_iso()
            atomic_json(checkpoint_path, checkpoint)
            batch_record["attempted_generation_ids"] = sorted(
                generation_id for generation_id in expected_batch_ids if generation_id in latest
            )
            batch_record["successful_generation_ids"] = sorted(
                generation_id for generation_id in expected_batch_ids
                if latest.get(generation_id, {}).get("generation_status") == "SUCCESS"
            )
            batch_record["failed_generation_ids"] = sorted(
                generation_id for generation_id in expected_batch_ids
                if generation_id in latest
                and latest[generation_id].get("generation_status") != "SUCCESS"
            )
            batch_record["updated_at"] = production._now_iso()
            atomic_json(_batch_record_path(output_dir, batch_id), batch_record)
            print(
                f"{generation_id} {status}",
                flush=True,
            )
    model.cpu()
    del model
    del tokenizer

    manifest = load_json(output_dir / "generations.json")
    success = sum(
        generation.get("generation_status") == "SUCCESS"
        for generation in manifest["generations"]
    )
    batch_record["attempted_generation_ids"] = sorted(
        generation_id for generation_id in expected_batch_ids if generation_id in latest
    )
    batch_record["successful_generation_ids"] = sorted(
        generation_id for generation_id in expected_batch_ids
        if latest.get(generation_id, {}).get("generation_status") == "SUCCESS"
    )
    batch_record["failed_generation_ids"] = sorted(
        generation_id for generation_id in expected_batch_ids
        if generation_id in latest
        and latest[generation_id].get("generation_status") != "SUCCESS"
    )
    batch_complete = len(batch_record["successful_generation_ids"]) == len(
        expected_batch_ids
    )
    batch_record["status"] = "completed" if batch_complete else "incomplete"
    batch_record["completed_at"] = _now_iso() if batch_complete else None
    batch_record["updated_at"] = _now_iso()
    atomic_json(_batch_record_path(output_dir, batch_id), batch_record)
    return {
        "status": "complete" if batch_complete else "incomplete",
        "batch_id": batch_record["batch_id"],
        "prompt_count": len(selected_batch),
        "batch_generation_count": len(expected_batch_ids),
        "batch_successful_generations": len(batch_record["successful_generation_ids"]),
        "batch_failed_generations": len(batch_record["failed_generation_ids"]),
        "batch_missing_generations": (
            len(expected_batch_ids) - len(batch_record["attempted_generation_ids"])
        ),
        "generation_count": len(manifest["generations"]),
        "successful_generations": success,
        "failed_generations": sum(
            generation.get("generation_status") != "SUCCESS"
            for generation in manifest["generations"]
        ),
        "missing_generation_count": (
            len(selection["selected_scenarios"]) * len(SEEDS)
            - len(manifest["generations"])
        ),
    }


def build_predictor_features(output_dir: Path) -> Dict[str, Any]:
    selection = load_json(output_dir / "selection.json")
    manifest = load_json(output_dir / "generations.json")
    if manifest.get("cohort_selection", {}).get(
        "selection_artifact_sha256"
    ) != selection_digest(selection):
        raise ValueError("Selection does not match the generation manifest")
    generations_by_prompt: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for generation in manifest.get("generations", []):
        generations_by_prompt[generation["prompt_id"]].append(generation)
    expected = len(selection["selected_scenarios"]) * len(SEEDS)
    if len(manifest.get("generations", [])) != expected or any(
        generation.get("generation_status") != "SUCCESS"
        for generation in manifest.get("generations", [])
    ):
        raise ValueError("Build features only after all cohort generations succeed")
    for row in selection["selected_scenarios"]:
        prompt_generations = generations_by_prompt[row["prompt_id"]]
        if (
            len(prompt_generations) != len(SEEDS)
            or {generation.get("seed") for generation in prompt_generations}
            != set(SEEDS)
        ):
            raise ValueError("Every selected prompt must have exactly ten generations")

    canonical_scenarios = load_json(CANONICAL_PATH)
    import run_production as production

    tokenizer = production.load_tokenizer()

    def token_count(text: str) -> int:
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])

    artifact = create_predictor_feature_artifact(
        selection,
        canonical_scenarios,
        output_dir / "hidden_states",
        token_counter=token_count,
    )
    artifact["pipeline"]["created_at"] = production._now_iso()
    feature_path = output_dir / "predictor_features.json"
    if feature_path.exists():
        existing = load_json(feature_path)
        if existing.get("predictor_features") != artifact["predictor_features"]:
            raise ValueError(
                f"Refusing to overwrite a different predictor feature artifact: {feature_path}"
            )
        return {"prompt_feature_count": len(artifact["predictor_features"]), "path": str(feature_path)}
    atomic_json(feature_path, artifact)
    return {
        "prompt_feature_count": len(artifact["predictor_features"]),
        "path": str(feature_path),
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--target-prompts", type=int, default=TARGET_PROMPTS)
    preflight.add_argument("--seed", type=int, default=SELECTION_SEED)
    preflight.add_argument("--batch-id", type=int, required=True)
    generate = subparsers.add_parser("generate")
    generate.add_argument("--batch-id", type=int, required=True)
    generate.add_argument("--max-new-prompts", type=int)
    subparsers.add_parser("build-features")
    args = parser.parse_args(argv)

    if args.command in ("preflight", "generate"):
        import run_production

        run_production._load_env_file()
        from src.evaluation.gemini_evaluator import discover_api_credentials

        credential_slots = [
            slot for slot, _ in discover_api_credentials()
        ]
        try:
            import google.genai  # noqa: F401

            genai_sdk_available = True
        except ImportError:
            genai_sdk_available = False
        if args.command == "preflight" and (
            args.target_prompts != TARGET_PROMPTS or args.seed != SELECTION_SEED
        ):
            raise ValueError("Batch preflight must use the saved 500-prompt selection")
        report = run_batch_preflight(
            args.output_dir,
            args.batch_id,
            credential_slots,
            genai_sdk_available,
        )
        if not report["can_execute"]:
            print("Batch stopped before generation: preflight blockers are present.")
            return 2
    if args.command == "preflight":
        return 0
    elif args.command == "generate":
        print(json.dumps(
            run_generation(args.output_dir, args.max_new_prompts, args.batch_id),
            indent=2,
        ))
    else:
        print(json.dumps(build_predictor_features(args.output_dir), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
