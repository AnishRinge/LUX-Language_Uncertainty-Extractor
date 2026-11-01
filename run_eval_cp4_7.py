#!/usr/bin/env python
"""Run or resume isolated Gemini 3.5 Flash-Lite CP4.7.0 evaluations."""

import argparse
import copy
import json
import sys
from datetime import timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_eval_cp4_5 as runner
from src.evaluation.prompt_cp4_7 import (
    CHANGE_REASON,
    EVALUATOR_MODEL,
    PROMPT_VERSION,
    build_messages,
    response_schema,
)
from src.evaluation.retry import RetryPolicy
from run_expanded_cohort import (
    OUTPUT_DIR as EXPANDED_OUTPUT_DIR,
    _batch_record_path,
    _batch_generation_ids,
    atomic_json,
)
from src.production.expanded_cohort import (
    BATCH_SIZE,
    TARGET_PROMPTS,
    batch_scenarios,
    load_json,
    selection_digest,
)

VERSION_DIR = ROOT / "data" / "production" / "cp4_7_0_gemini_3_5_flash_lite"
EVALUATIONS_DIR = VERSION_DIR / "evaluations"
RUN_LOG_PATH = VERSION_DIR / "evaluation_run.json"
TARGETS_PATH = VERSION_DIR / "targets.json"


def _batch_evaluation_input(generation_path: Path, batch_id: int) -> Path:
    selection_path = EXPANDED_OUTPUT_DIR / "selection.json"
    selection = load_json(selection_path)
    rows = selection.get("selected_scenarios", [])
    if (
        len(rows) != TARGET_PROMPTS
        or selection.get("target_prompts") != TARGET_PROMPTS
        or selection.get("seeds") != list(range(1001, 1011))
    ):
        raise ValueError("Evaluation requires the fixed 500-prompt selection")
    batch_scenarios(selection, batch_id)
    batch_rows = rows[: batch_id * BATCH_SIZE]
    allowed_prompt_ids = {row["prompt_id"] for row in batch_rows}
    expected_generation_ids = _batch_generation_ids(batch_rows)

    manifest = load_json(generation_path)
    if manifest.get("cohort_selection", {}).get(
        "selection_artifact_sha256"
    ) != selection_digest(selection):
        raise ValueError("Generation manifest does not match the saved 500-prompt selection")
    generations = [
        generation
        for generation in manifest.get("generations", [])
        if generation.get("prompt_id") in allowed_prompt_ids
    ]
    by_id = {generation.get("generation_id"): generation for generation in generations}
    if len(by_id) != len(generations) or set(by_id) != set(expected_generation_ids):
        raise ValueError(
            f"Batch {batch_id} does not have all expected generation records; "
            "finish or repair generation before evaluation"
        )
    if any(generation.get("generation_status") != "SUCCESS" for generation in generations):
        raise ValueError("Batch evaluation requires successful generations only")

    if batch_id > 1:
        for previous_id in range(1, batch_id):
            previous = load_json(_batch_record_path(EXPANDED_OUTPUT_DIR, previous_id))
            if previous.get("status") != "completed":
                raise ValueError(
                    f"Batch {previous_id} must complete before evaluating batch {batch_id}"
                )
    current_record = load_json(_batch_record_path(EXPANDED_OUTPUT_DIR, batch_id))
    if current_record.get("status") != "completed":
        raise ValueError(f"Batch {batch_id} generation is not marked completed")

    evaluation_input = EXPANDED_OUTPUT_DIR / "batches" / (
        f"batch_{batch_id:02d}_evaluation_input.json"
    )
    batch_manifest = copy.deepcopy(manifest)
    batch_manifest["generations"] = generations
    batch_manifest["scenarios"] = [
        {"dataset_index": row["dataset_index"], "sample_id": row["prompt_id"]}
        for row in batch_rows
    ]
    atomic_json(evaluation_input, batch_manifest)
    return evaluation_input


def _update_batch_evaluation_status(batch_id: int) -> dict:
    path = _batch_record_path(EXPANDED_OUTPUT_DIR, batch_id)
    batch = load_json(path)
    generation_ids = batch["expected_generation_ids"]
    statuses = {}
    for generation_id in generation_ids:
        record_path = EVALUATIONS_DIR / runner._safe_generation_name(generation_id)
        candidates = [record_path]
        retry_dir = EVALUATIONS_DIR / "retries"
        if retry_dir.exists():
            candidates.extend(
                sorted(retry_dir.glob(record_path.stem + ".retry_*.json"))
            )
        records = [load_json(candidate) for candidate in candidates if candidate.exists()]
        if records:
            record = max(records, key=lambda item: item.get("timestamp", ""))
            if (
                record.get("source_generation_id") != generation_id
                or record.get("evaluator_model") != EVALUATOR_MODEL
                or record.get("prompt_version") != PROMPT_VERSION
            ):
                raise ValueError(
                    f"Evaluation record has mismatched provenance: {generation_id}"
                )
            statuses[generation_id] = record.get("evaluation_status", "UNKNOWN")
    batch["evaluation_status_counts"] = {
        status: sum(value == status for value in statuses.values())
        for status in sorted(set(statuses.values()))
    }
    batch["evaluation_completed_generation_ids"] = sorted(
        generation_id for generation_id, status in statuses.items()
        if status in ("SUCCESS", "MANUAL_REVIEW")
    )
    batch["evaluation_failed_generation_ids"] = sorted(
        generation_id for generation_id, status in statuses.items()
        if status == "FAILED"
    )
    batch["evaluation_remaining_generation_ids"] = sorted(
        set(generation_ids) - set(statuses)
    )
    batch["evaluation_status"] = (
        "completed"
        if len(batch["evaluation_completed_generation_ids"]) == len(generation_ids)
        else "incomplete"
    )
    from run_production import _now_iso

    batch["evaluation_updated_at"] = _now_iso()
    atomic_json(path, batch)
    return batch


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--batch-id", type=int, required=True)
    evaluate.add_argument("--spacing", type=float, default=5.0)
    evaluate.add_argument("--max-seconds", type=int, default=9000)
    evaluate.add_argument("--max-evaluations", type=int)
    evaluate.add_argument("--quota-cooldown-seconds", type=int, default=60)
    evaluate.add_argument(
        "--generation-path", type=Path,
        default=EXPANDED_OUTPUT_DIR / "generations.json",
    )
    evaluate.add_argument(
        "--single-attempt",
        action="store_true",
        help="Use only the primary credential and disable retries for one controlled request",
    )
    target = subparsers.add_parser("build-targets")
    target.add_argument("--target-path", type=Path, default=TARGETS_PATH)
    target.add_argument(
        "--generation-path", type=Path, default=runner.GENERATION_PATH
    )
    args = parser.parse_args(argv)

    if args.command == "evaluate":
        import run_production

        run_production._load_env_file()
        batch_generation_path = _batch_evaluation_input(
            args.generation_path, args.batch_id
        )
        result = runner.run_evaluation(
            generation_path=batch_generation_path,
            evaluations_dir=EVALUATIONS_DIR,
            log_path=RUN_LOG_PATH,
            spacing_seconds=args.spacing,
            max_seconds=args.max_seconds,
            evaluator_model=EVALUATOR_MODEL,
            prompt_version=PROMPT_VERSION,
            change_reason=CHANGE_REASON,
            message_builder=build_messages,
            response_schema_builder=response_schema,
            max_evaluations=args.max_evaluations,
            quota_cooldown=timedelta(seconds=args.quota_cooldown_seconds),
            evaluator_options=(
                {
                    "max_credential_attempts": 1,
                    "retry_policy": RetryPolicy(max_retries=0),
                }
                if args.single_attempt
                else None
            ),
        )
        batch = _update_batch_evaluation_status(args.batch_id)
        print(json.dumps({
            "runner": result,
            "batch_evaluation_status": batch["evaluation_status"],
            "batch_evaluation_status_counts": batch["evaluation_status_counts"],
            "failed_evaluations": len(batch["evaluation_failed_generation_ids"]),
            "remaining_evaluations": len(batch["evaluation_remaining_generation_ids"]),
        }, indent=2))
        return (
            0
            if result["stop_reason"] == "completed"
            and batch["evaluation_status"] == "completed"
            else 2
        )

    result = runner.build_targets(
        generation_path=args.generation_path,
        evaluations_dir=EVALUATIONS_DIR,
        targets_path=args.target_path,
        evaluator_model=EVALUATOR_MODEL,
        prompt_version=PROMPT_VERSION,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
