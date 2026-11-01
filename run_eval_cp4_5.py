#!/usr/bin/env python
"""Run and resume the isolated CP4.5.0 Gemini evaluation dataset."""

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.gemini_evaluator import (
    EvaluationRequest,
    EvaluatorConfigError,
    GeminiEvaluator,
    GlobalModelConfigurationError,
    QuotaExhaustedError,
    discover_api_credentials,
)
from src.evaluation.prompt_cp4_5 import (
    CHANGE_REASON,
    EVALUATOR_MODEL,
    PROMPT_VERSION,
    build_messages,
    response_schema,
)
from src.evaluation.schemas import (
    SCHEMA_VERSION,
    EvaluationRecord,
    EvaluationStatus,
    GenerationLabel,
)

GENERATION_PATH = ROOT / "data" / "production" / "generations.json"
CANONICAL_SCENARIOS_PATH = (
    ROOT / "data" / "prompts" / "squad_v2_canonical_scenarios.json"
)
VERSION_DIR = ROOT / "data" / "production" / "cp4_5_0_gemini_3_8_flash"
EVALUATIONS_DIR = VERSION_DIR / "evaluations"
RUN_LOG_PATH = VERSION_DIR / "evaluation_run.json"
TARGETS_PATH = VERSION_DIR / "targets.json"
PREDICTOR_REPORT_PATH = VERSION_DIR / "predictor_report.json"
FEATURES_PATH = ROOT / "data" / "production" / "predictor_features.json"
HIDDEN_STATES_DIR = ROOT / "data" / "production" / "hidden_states"
MODEL_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
REQUIRED_GENERATIONS_PER_PROMPT = 10
QUOTA_COOLDOWN = timedelta(hours=24)
TARGET_LABELS = {GenerationLabel.RELIABLE.value, GenerationLabel.UNRELIABLE.value}
GLOBAL_ERROR_TYPES = {
    "GlobalModelConfigurationError",
    "EvaluatorConfigError",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _safe_generation_name(generation_id: str) -> str:
    return generation_id.replace(":", "_").replace("/", "_") + ".json"


def _wilson_interval(successes: int, total: int) -> Optional[List[float]]:
    """Return a two-sided 95% Wilson interval for a repeated-generation rate."""
    if total <= 0:
        return None
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1 - proportion) / total
            + z * z / (4 * total * total)
        )
        / denominator
    )
    return [max(0.0, center - margin), min(1.0, center + margin)]


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _new_log(
    generation_count: int,
    prompt_count: int,
    evaluator_model: str = EVALUATOR_MODEL,
    prompt_version: str = PROMPT_VERSION,
    change_reason: str = CHANGE_REASON,
) -> Dict[str, Any]:
    return {
        "evaluator_model": evaluator_model,
        "prompt_version": prompt_version,
        "schema_version": SCHEMA_VERSION,
        "change_reason": change_reason,
        "generation_count": generation_count,
        "prompt_count": prompt_count,
        "started_at": _now(),
        "updated_at": _now(),
        "iterations": [],
        "unusable_credential_slots": {},
        "quota_exhausted_at": {},
        "stop_reason": None,
    }


def _reopen_expired_quota_slots(
    log: Dict[str, Any],
    now: Optional[datetime] = None,
    cooldown: timedelta = QUOTA_COOLDOWN,
) -> List[str]:
    """Re-enable quota-blocked projects after a conservative 24-hour cooldown."""
    current = now or datetime.now(timezone.utc)
    blocked_at = log.setdefault("quota_exhausted_at", {})
    unusable = log.setdefault("unusable_credential_slots", {})
    reopened = []
    for slot, timestamp in list(blocked_at.items()):
        try:
            exhausted = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        except (AttributeError, ValueError):
            continue
        if exhausted.tzinfo is None:
            exhausted = exhausted.replace(tzinfo=timezone.utc)
        if current - exhausted >= cooldown:
            if unusable.get(slot) == "QuotaExhaustedError":
                unusable.pop(slot)
                reopened.append(slot)
            blocked_at.pop(slot, None)
    return reopened


def _restore_quota_timestamps(log: Dict[str, Any], evaluations_dir: Path) -> None:
    """Backfill cooldown timestamps from persisted records after older runs."""
    blocked_at = log.setdefault("quota_exhausted_at", {})
    unusable = log.setdefault("unusable_credential_slots", {})
    missing = {
        slot
        for slot, reason in unusable.items()
        if reason == "QuotaExhaustedError" and slot not in blocked_at
    }
    if not missing:
        return
    latest: Dict[str, str] = {}
    for path in sorted(evaluations_dir.glob("*.json")):
        try:
            record = EvaluationRecord.from_dict(_load_json(path))
        except (OSError, ValueError, KeyError):
            continue
        for attempt in record.attempts:
            slot = attempt.get("key_slot")
            if (
                slot in missing
                and attempt.get("error_type") == "QuotaExhaustedError"
            ):
                previous = latest.get(slot)
                if previous is None or record.timestamp > previous:
                    latest[slot] = record.timestamp
    blocked_at.update(latest)


def _failed_from_exception(
    request,
    error: Exception,
    attempts: List[Dict[str, Any]],
    error_message: str,
    evaluator_model: str = EVALUATOR_MODEL,
    prompt_version: str = PROMPT_VERSION,
) -> EvaluationRecord:
    record = EvaluationRecord(
        schema_version=SCHEMA_VERSION,
        evaluator_model=evaluator_model,
        prompt_version=prompt_version,
        timestamp=_now(),
        source_generation_id=request.source_generation_id,
        dataset_id=request.dataset_id,
        evaluation_status=EvaluationStatus.FAILED,
        retry_count=max(0, len(attempts) - 1),
        claims=[],
        generation_label=None,
        generation_failure_types=[],
        rationale="",
        needs_human_review=False,
        error_type=type(error).__name__,
        error_message=error_message,
        attempts=attempts,
    )
    return record.validate()


def _load_matching_record(
    path: Path,
    generation_id: str,
    evaluator_model: str = EVALUATOR_MODEL,
    prompt_version: str = PROMPT_VERSION,
) -> Optional[EvaluationRecord]:
    if not path.exists():
        return None
    record = EvaluationRecord.from_dict(_load_json(path))
    expected = (evaluator_model, prompt_version, generation_id)
    actual = (
        record.evaluator_model,
        record.prompt_version,
        record.source_generation_id,
    )
    if actual != expected:
        raise ValueError(
            "Versioned evaluation path contains mismatched provenance: "
            f"{path} expected={expected!r} actual={actual!r}"
        )
    return record


def _latest_generation_record(
    evaluations_dir: Path,
    generation_id: str,
    evaluator_model: str = EVALUATOR_MODEL,
    prompt_version: str = PROMPT_VERSION,
) -> Optional[EvaluationRecord]:
    safe_name = _safe_generation_name(generation_id)
    safe_stem = Path(safe_name).stem
    retry_dir = evaluations_dir / "retries"
    candidates = [evaluations_dir / safe_name]
    if retry_dir.exists():
        candidates.extend(sorted(retry_dir.glob(f"{safe_stem}.retry_*.json")))
    records = [
        _load_matching_record(path, generation_id, evaluator_model, prompt_version)
        for path in candidates
        if path.exists()
    ]
    return records[-1] if records else None


def _next_retry_path(evaluations_dir: Path, generation_id: str) -> Path:
    safe_stem = Path(_safe_generation_name(generation_id)).stem
    retry_dir = evaluations_dir / "retries"
    retry_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(retry_dir.glob(f"{safe_stem}.retry_*.json"))
    numbers = []
    for path in paths:
        try:
            numbers.append(int(path.stem.rsplit(".retry_", 1)[1]))
        except (IndexError, ValueError):
            continue
    return retry_dir / f"{safe_stem}.retry_{max(numbers, default=0) + 1:03d}.json"


def run_evaluation(
    generation_path: Path = GENERATION_PATH,
    scenarios_path: Path = CANONICAL_SCENARIOS_PATH,
    evaluations_dir: Path = EVALUATIONS_DIR,
    log_path: Path = RUN_LOG_PATH,
    *,
    spacing_seconds: float = 1.0,
    max_seconds: int = 9000,
    evaluator_class: Type[GeminiEvaluator] = GeminiEvaluator,
    evaluator_model: str = EVALUATOR_MODEL,
    prompt_version: str = PROMPT_VERSION,
    change_reason: str = CHANGE_REASON,
    message_builder=build_messages,
    response_schema_builder=response_schema,
    max_evaluations: Optional[int] = None,
    evaluator_options: Optional[Dict[str, Any]] = None,
    quota_cooldown: timedelta = QUOTA_COOLDOWN,
) -> Dict[str, Any]:
    """Evaluate every existing successful generation, resuming versioned files."""
    artifact = _load_json(generation_path)
    model_info = artifact.get("model", {})
    if model_info.get("name") != "Qwen/Qwen3-1.7B":
        raise ValueError("Generation manifest does not contain the frozen Qwen target")
    if model_info.get("revision") != MODEL_REVISION:
        raise ValueError("Generation manifest target revision differs from frozen revision")

    generations = artifact.get("generations", [])
    if not generations:
        raise ValueError("Generation manifest contains no generations")
    generation_ids = [g.get("generation_id") for g in generations]
    if None in generation_ids or len(set(generation_ids)) != len(generation_ids):
        raise ValueError("Generation manifest has missing or duplicate generation IDs")
    if any(g.get("generation_status") != "SUCCESS" for g in generations):
        raise ValueError("Manifest includes non-successful generations; refusing partial input")

    scenarios = _load_json(scenarios_path)
    configured_slots = {slot for slot, _ in discover_api_credentials()}
    existing_log = _load_json(log_path) if log_path.exists() else None
    cohort_generation_ids_sha256 = hashlib.sha256(
        json.dumps(sorted(generation_ids), separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if existing_log and (
        existing_log.get("evaluator_model") != evaluator_model
        or existing_log.get("prompt_version") != prompt_version
    ):
        raise ValueError("Run log belongs to a different evaluator version")
    if existing_log:
        stored_cohort_hash = existing_log.get("cohort_generation_ids_sha256")
        logged_generation_ids = {
            iteration.get("generation_id")
            for iteration in existing_log.get("iterations", [])
            if iteration.get("generation_id")
        }
        if stored_cohort_hash and stored_cohort_hash != cohort_generation_ids_sha256 and (
            not logged_generation_ids
            or not logged_generation_ids.issubset(set(generation_ids))
            or len(generation_ids) <= len(logged_generation_ids)
        ):
            raise ValueError("Run log belongs to a different generation cohort")
        if logged_generation_ids and not logged_generation_ids.issubset(set(generation_ids)):
            raise ValueError(
                "Generation manifest omits generation IDs already recorded in this run log"
            )
    log = existing_log or _new_log(
        len(generations),
        len({g["prompt_id"] for g in generations}),
        evaluator_model,
        prompt_version,
        change_reason,
    )
    log.setdefault("iterations", [])
    log.setdefault("unusable_credential_slots", {})
    log.setdefault("quota_exhausted_at", {})
    _restore_quota_timestamps(log, evaluations_dir)
    reopened = _reopen_expired_quota_slots(log, cooldown=quota_cooldown)
    log["started_at"] = _now()
    log["reopened_quota_slots"] = reopened
    log["generation_count"] = len(generations)
    log["prompt_count"] = len({g["prompt_id"] for g in generations})
    log["cohort_generation_ids_sha256"] = cohort_generation_ids_sha256
    log["total_generations"] = len(generations)
    log["updated_at"] = _now()
    evaluations_dir.mkdir(parents=True, exist_ok=True)

    evaluator_arguments = {
        "model": evaluator_model,
        "prompt_version": prompt_version,
        "excluded_credential_slots": log["unusable_credential_slots"].keys(),
        "message_builder": message_builder,
        "response_schema_builder": response_schema_builder,
    }
    evaluator_arguments.update(evaluator_options or {})
    evaluator = evaluator_class(**evaluator_arguments)
    stop_reason = None
    start_time = time.monotonic()
    evaluated_count = 0
    try:
        for index, generation in enumerate(generations, start=1):
            if time.monotonic() - start_time >= max_seconds:
                stop_reason = "execution_budget_reached"
                break

            generation_id = generation["generation_id"]
            existing = _latest_generation_record(
                evaluations_dir, generation_id, evaluator_model, prompt_version
            )
            if existing is not None:
                if existing.evaluation_status in (
                    EvaluationStatus.SUCCESS,
                    EvaluationStatus.MANUAL_REVIEW,
                ):
                    continue
                if existing.error_type != "QuotaExhaustedError":
                    continue
                attempt_slots = {
                    attempt.get("key_slot")
                    for attempt in existing.attempts
                    if attempt.get("key_slot")
                }
                slot_still_blocked = any(
                    slot in log["unusable_credential_slots"]
                    for slot in attempt_slots
                )
                try:
                    failure_time = datetime.fromisoformat(
                        existing.timestamp.replace("Z", "+00:00")
                    )
                    if failure_time.tzinfo is None:
                        failure_time = failure_time.replace(tzinfo=timezone.utc)
                    cooling_down = (
                        datetime.now(timezone.utc) - failure_time < quota_cooldown
                    )
                except ValueError:
                    cooling_down = True
                if slot_still_blocked or cooling_down:
                    continue

            if configured_slots and configured_slots.issubset(
                log["unusable_credential_slots"]
            ):
                stop_reason = "all_configured_credentials_unusable"
                break

            dataset_index = generation["dataset_index"]
            if dataset_index < 0 or dataset_index >= len(scenarios):
                raise ValueError(
                    f"Generation {generation_id} has invalid dataset index {dataset_index}"
                )
            scenario = scenarios[dataset_index]
            request = EvaluationRequest(
                question=generation.get("question", ""),
                context=generation.get("context", ""),
                generated_answer=generation.get("generated_answer", ""),
                source_generation_id=generation_id,
                dataset_id=str(
                    generation.get(
                        "dataset_source",
                        scenario.get("source", {}).get("dataset", "squad_v2"),
                    )
                ),
                sample_id=generation.get("sample_id", ""),
                generation_index=generation.get("generation_index", 0),
                seed=generation.get("seed"),
                model_name=generation.get("model_name", ""),
                model_revision=generation.get("model_revision", ""),
                answerability=generation.get("answerability", ""),
            )
            if (
                not request.question
                or not request.context
                or not request.generated_answer
            ):
                raise ValueError(
                    f"Manifest prompt or answer is missing for {generation_id}"
                )

            error: Optional[Exception] = None
            try:
                record = evaluator.evaluate(request)
            except QuotaExhaustedError as exc:
                error = exc
                record = _failed_from_exception(
                    request,
                    exc,
                    getattr(exc, "attempts", []),
                    evaluator._redact(str(exc)),
                    evaluator_model,
                    prompt_version,
                )
            record.validate()
            if (
                record.source_generation_id != generation_id
                or record.evaluator_model != evaluator_model
                or record.prompt_version != prompt_version
            ):
                raise ValueError("Evaluator returned mismatched generation/version provenance")
            if (
                record.evaluation_status == EvaluationStatus.FAILED
                and record.generation_label is not None
            ):
                raise ValueError("Failed evaluation cannot carry a generation label")
            attempts_this_evaluation = list(record.attempts)
            if existing is not None:
                combined_attempts = list(existing.attempts) + list(record.attempts)
                for attempt_number, attempt in enumerate(combined_attempts, start=1):
                    attempt["attempt"] = attempt_number
                record.attempts = combined_attempts
                record.retry_count = max(0, len(combined_attempts) - 1)
                record.validate()
                output_path = _next_retry_path(evaluations_dir, generation_id)
            else:
                output_path = evaluations_dir / _safe_generation_name(generation_id)
            _atomic_json(output_path, record.to_dict())

            for attempt in attempts_this_evaluation:
                slot = attempt.get("key_slot")
                error_type = attempt.get("error_type")
                if slot and error_type in {
                    "CredentialAccessError",
                    "QuotaExhaustedError",
                }:
                    log["unusable_credential_slots"][slot] = error_type
                    if error_type == "QuotaExhaustedError":
                        log["quota_exhausted_at"][slot] = _now()
            evaluator.exclude_credential_slots(
                log["unusable_credential_slots"].keys()
            )

            iteration = {
                "timestamp": _now(),
                "generation_id": generation_id,
                "evaluation_status": record.evaluation_status.value,
                "generation_label": (
                    record.generation_label.value
                    if record.generation_label is not None
                    else None
                ),
                "error_type": record.error_type,
                "attempts": len(attempts_this_evaluation),
                "credential_slots": sorted({
                    a.get("key_slot")
                    for a in attempts_this_evaluation
                    if a.get("key_slot")
                }),
                "exit_code": 0,
            }
            log["iterations"].append(iteration)
            evaluated_count += 1
            log["updated_at"] = _now()
            log["unusable_credential_slots"] = dict(
                log["unusable_credential_slots"]
            )
            _atomic_json(log_path, log)
            print(
                f"[{index}/{len(generations)}] {generation_id} "
                f"{record.evaluation_status.value} "
                f"{record.generation_label.value if record.generation_label else '-'}",
                flush=True,
            )

            if isinstance(error, GlobalModelConfigurationError) or (
                record.error_type in GLOBAL_ERROR_TYPES
            ):
                stop_reason = "global_model_configuration_error"
                break
            if max_evaluations is not None and evaluated_count >= max_evaluations:
                stop_reason = "evaluation_limit_reached"
                break
            if configured_slots and configured_slots.issubset(
                log["unusable_credential_slots"]
            ):
                stop_reason = "all_configured_credentials_unusable"
                break
            if spacing_seconds > 0 and index < len(generations):
                time.sleep(spacing_seconds)
    finally:
        evaluator.close()

    counts = _evaluation_counts(evaluations_dir)
    completed = sum(counts["statuses"].values())
    if stop_reason is None:
        stop_reason = "completed" if completed == len(generations) else "interrupted"
    log.update({
        "updated_at": _now(),
        "stop_reason": stop_reason,
        "elapsed_seconds": round(time.monotonic() - start_time, 2),
        "completed_record_count": completed,
        "remaining_without_record": len(generations) - completed,
        "status_counts": counts["statuses"],
        "label_counts": counts["labels"],
    })
    _atomic_json(log_path, log)
    print(json.dumps(log, indent=2, ensure_ascii=False), flush=True)
    return log


def _evaluation_counts(evaluations_dir: Path) -> Dict[str, Counter]:
    statuses: Counter = Counter()
    labels: Counter = Counter()
    if evaluations_dir.exists():
        paths = list(evaluations_dir.glob("*.json"))
        retry_dir = evaluations_dir / "retries"
        if retry_dir.exists():
            paths.extend(retry_dir.glob("*.json"))
        latest_by_generation: Dict[str, EvaluationRecord] = {}
        for path in sorted(paths):
            record = EvaluationRecord.from_dict(_load_json(path))
            previous = latest_by_generation.get(record.source_generation_id)
            if previous is None or record.timestamp >= previous.timestamp:
                latest_by_generation[record.source_generation_id] = record
        for record in latest_by_generation.values():
            statuses[record.evaluation_status.value] += 1
            if record.generation_label is not None:
                labels[record.generation_label.value] += 1
    return {"statuses": statuses, "labels": labels}


def build_targets(
    generation_path: Path = GENERATION_PATH,
    evaluations_dir: Path = EVALUATIONS_DIR,
    targets_path: Path = TARGETS_PATH,
    evaluator_model: str = EVALUATOR_MODEL,
    prompt_version: str = PROMPT_VERSION,
) -> Dict[str, Any]:
    """Build strict ten-generation empirical risks from this evaluator version."""
    generations = _load_json(generation_path)["generations"]
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for generation in generations:
        groups[generation["prompt_id"]].append(generation)

    target_rows: List[Dict[str, Any]] = []
    prompt_distributions: Counter = Counter()
    aggregate_labels: Counter = Counter()
    risk_distribution: Counter = Counter()
    for prompt_id, prompt_generations in sorted(groups.items()):
        outcomes: Counter = Counter()
        binary_labels: List[str] = []
        for generation in prompt_generations:
            record = _latest_generation_record(
                evaluations_dir,
                generation["generation_id"],
                evaluator_model,
                prompt_version,
            )
            if record is None:
                outcomes["MISSING"] += 1
                continue
            status = record.evaluation_status.value
            label = record.generation_label.value if record.generation_label else None
            if label:
                outcomes[label] += 1
                aggregate_labels[label] += 1
                if status == EvaluationStatus.SUCCESS.value and label in TARGET_LABELS:
                    binary_labels.append(label)
            if status != EvaluationStatus.SUCCESS.value:
                outcomes[status] += 1

        expected = len(prompt_generations) == REQUIRED_GENERATIONS_PER_PROMPT
        complete = expected and len(binary_labels) == REQUIRED_GENERATIONS_PER_PROMPT
        prompt_distributions[
            "complete_binary_target" if complete else "incomplete_or_inadequate"
        ] += 1
        risk = (
            outcomes[GenerationLabel.UNRELIABLE.value]
            / REQUIRED_GENERATIONS_PER_PROMPT
            if complete
            else None
        )
        if risk is not None:
            risk_distribution[f"{risk:.1f}"] += 1
        target_rows.append({
            "prompt_id": prompt_id,
            "dataset_index": prompt_generations[0]["dataset_index"],
            "n_generations": len(prompt_generations),
            "n_reliable": outcomes[GenerationLabel.RELIABLE.value],
            "n_unreliable": outcomes[GenerationLabel.UNRELIABLE.value],
            "n_inadequate": outcomes[GenerationLabel.INADEQUATE.value],
            "n_manual_review": outcomes[EvaluationStatus.MANUAL_REVIEW.value],
            "n_failed": outcomes[EvaluationStatus.FAILED.value],
            "n_missing": outcomes["MISSING"],
            "empirical_hallucination_risk": risk,
            "risk_wilson_95_interval": (
                _wilson_interval(
                    outcomes[GenerationLabel.UNRELIABLE.value],
                    REQUIRED_GENERATIONS_PER_PROMPT,
                )
                if complete
                else None
            ),
            "target_eligible": complete,
        })

    artifact = {
        "evaluator_model": evaluator_model,
        "prompt_version": prompt_version,
        "risk_definition": (
            "UNRELIABLE / 10 repeated generations; risk is emitted only when "
            "all ten are SUCCESS with RELIABLE or UNRELIABLE labels"
        ),
        "inadequate_handling": (
            "INADEQUATE is reported but excluded; incomplete prompts receive no target"
        ),
        "n_prompts": len(target_rows),
        "prompt_target_counts": dict(prompt_distributions),
        "label_counts": dict(aggregate_labels),
        "risk_distribution": dict(risk_distribution),
        "targets": target_rows,
    }
    _atomic_json(targets_path, artifact)
    return artifact


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--spacing", type=float, default=1.0)
    evaluate_parser.add_argument("--max-seconds", type=int, default=9000)
    target_parser = subparsers.add_parser("build-targets")
    target_parser.add_argument("--target-path", type=Path, default=TARGETS_PATH)
    args = parser.parse_args(argv)

    if args.command == "evaluate":
        import run_production

        run_production._load_env_file()
        log = run_evaluation(
            spacing_seconds=args.spacing,
            max_seconds=args.max_seconds,
        )
        return 0 if log["stop_reason"] == "completed" else 2
    targets = build_targets(targets_path=args.target_path)
    print(json.dumps(targets, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
