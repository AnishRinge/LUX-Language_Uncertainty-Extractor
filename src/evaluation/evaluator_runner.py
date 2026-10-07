"""CP4.4 evaluator runner.

Orchestrates end-to-end evaluation of Qwen3 generations against the frozen
CP4.2 generation artifacts, persists reproducible evaluation records, and
computes the empirical hallucination-risk target.

Empirical target (frozen):
    empirical_hallucination_risk =
        UNRELIABLE generations / total evaluated generations

Per the frozen design, only evaluations with ``evaluation_status == SUCCESS``
("successfully resolved") contribute to the denominator; FAILED and
MANUAL_REVIEW evaluations are preserved with their reasons and never silently
discarded, and a FAILED result is never converted into a generation label.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .schemas import (
    SCHEMA_VERSION,
    EvaluationRecord,
    EvaluationStatus,
    GenerationLabel,
)
from .gemini_evaluator import EvaluationRequest, GeminiEvaluator

DEFAULT_EVALUATIONS_DIR = Path("data/evaluations")
CP4_2_5_ARTIFACT_PATH = Path("data/pilots/cp4_2_5_representative_validation.json")
CANONICAL_SCENARIOS_PATH = Path("data/prompts/squad_v2_canonical_scenarios.json")

# Pilot outputs are persisted separately from any future production results.
DEFAULT_PILOT_DIR = Path("data/pilots")
DEFAULT_PILOT_ARTIFACT = DEFAULT_PILOT_DIR / "cp4_4_evaluator_pilot.json"
DEFAULT_PILOT_REPORT = Path("docs/cp4_4_evaluator_pilot_report.md")


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    """Atomically write JSON so a reader never observes a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_cp4_2_5_generations(
    artifact_path: Path = CP4_2_5_ARTIFACT_PATH,
    canonical_path: Path = CANONICAL_SCENARIOS_PATH,
) -> Tuple[List[Dict[str, Any]], Dict[int, Dict[str, Any]]]:
    """Load the CP4.2.5 generations and their canonical scenarios."""
    artifact = _load_json(artifact_path)
    scenarios = _load_json(canonical_path)
    return artifact["generations"], scenarios


def generation_id(generation: Dict[str, Any]) -> str:
    """Stable identifier for a generation: sample_id:dataset_index:seed."""
    return "{0}:{1}:{2}".format(
        generation.get("sample_id", "unknown"),
        generation.get("dataset_index", -1),
        generation.get("seed", -1),
    )


def build_request_from_cp4_2_5(
    generation: Dict[str, Any], scenario: Dict[str, Any]
) -> EvaluationRequest:
    """Build an ``EvaluationRequest`` from a CP4.2.5 generation + scenario.

    The evaluator receives ONLY the question, the supplied context, and the
    generated answer. Decoding config, seed, hidden states, logits and the
    model identity are deliberately excluded from the request inputs (the
    ``source_generation_id`` / ``dataset_id`` are metadata for auditability,
    never evaluator input content).
    """
    question = scenario["prompt"]["question"]
    context = scenario["prompt"]["context"]
    generated_answer = generation.get("generated_answer", "")
    return EvaluationRequest(
        question=question,
        context=context,
        generated_answer=generated_answer,
        source_generation_id=generation_id(generation),
        dataset_id=str(scenario.get("source", {}).get("dataset", "squad_v2")),
        sample_id=generation.get("sample_id", ""),
        generation_index=generation.get("generation_index", 0),
        seed=generation.get("seed"),
        model_name=generation.get("model_name", ""),
        model_revision=generation.get("model_revision", ""),
        answerability=generation.get("answerability", ""),
    )


class EvaluationRunner:
    """Run an evaluator over generations and persist reproducible records."""

    def __init__(
        self,
        evaluator: GeminiEvaluator,
        output_dir: Path = DEFAULT_EVALUATIONS_DIR,
        pilot_mode: bool = False,
    ):
        self.evaluator = evaluator
        self.output_dir = Path(output_dir)
        self.pilot_mode = pilot_mode

    def _record_path(self, record: EvaluationRecord) -> Path:
        safe_id = record.source_generation_id.replace(":", "_").replace("/", "_")
        return self.output_dir / "{0}.json".format(safe_id)

    def run_one(self, request: EvaluationRequest) -> EvaluationRecord:
        record = self.evaluator.evaluate(request)
        self._persist(record)
        return record

    def run_many(self, requests: List[EvaluationRequest], batch_size: int = 0) -> List[EvaluationRecord]:
        """Evaluate a batch, persisting each record. ``batch_size==0`` => no pause."""
        results: List[EvaluationRecord] = []
        for index, request in enumerate(requests, start=1):
            results.append(self.run_one(request))
            if batch_size and index % batch_size == 0:
                # Gentle pacing; configurable rather than hard-coded throttling.
                import time as _time

                _time.sleep(0.0)
        return results

    def _persist(self, record: EvaluationRecord) -> Path:
        path = self._record_path(record)
        path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(path, record.to_dict())
        return path

    def load_all(self) -> List[EvaluationRecord]:
        records: List[EvaluationRecord] = []
        if not self.output_dir.exists():
            return records
        for path in sorted(self.output_dir.glob("*.json")):
            try:
                records.append(EvaluationRecord.from_dict(_load_json(path)))
            except ValueError:
                continue
        return records


def evaluate_generations(
    evaluator: GeminiEvaluator,
    generations: List[Dict[str, Any]],
    scenarios: Dict[int, Dict[str, Any]],
    output_dir: Path = DEFAULT_EVALUATIONS_DIR,
    pilot_mode: bool = False,
) -> List[EvaluationRecord]:
    """Convenience: build requests from CP4.2.5 generations and evaluate them."""
    runner = EvaluationRunner(evaluator, output_dir=output_dir, pilot_mode=pilot_mode)
    requests: List[EvaluationRequest] = []
    for generation in generations:
        scenario = scenarios[generation["dataset_index"]]
        requests.append(build_request_from_cp4_2_5(generation, scenario))
    return runner.run_many(requests)


def compute_empirical_risk(records: List[EvaluationRecord]) -> Dict[str, Any]:
    """Compute the frozen empirical hallucination-risk target.

    empirical_hallucination_risk = UNRELIABLE / total successfully-resolved

    Only ``evaluation_status == SUCCESS`` records count toward the denominator
    ("successfully resolved generation evaluations may contribute to the
    empirical target"). FAILED and MANUAL_REVIEW records are reported in full
    but excluded from the rate.
    """
    by_status: Dict[str, int] = {s.value: 0 for s in EvaluationStatus}
    by_label: Dict[str, int] = {l.value: 0 for l in GenerationLabel}
    resolved = [r for r in records if r.evaluation_status == EvaluationStatus.SUCCESS]
    for r in records:
        by_status[r.evaluation_status.value] = by_status.get(r.evaluation_status.value, 0) + 1
    for r in resolved:
        if r.generation_label is not None:
            by_label[r.generation_label.value] = by_label.get(r.generation_label.value, 0) + 1
    total_resolved = len(resolved)
    unreliable = by_label.get(GenerationLabel.UNRELIABLE.value, 0)
    risk = (unreliable / total_resolved) if total_resolved else None
    return {
        "empirical_hallucination_risk": risk,
        "total_resolved": total_resolved,
        "unreliable_resolved": unreliable,
        "status_counts": by_status,
        "label_counts": by_label,
        "included_in_denominator": "evaluation_status == SUCCESS only",
    }


def aggregate_pilot_report(records: List[EvaluationRecord]) -> Dict[str, Any]:
    """Summarise a pilot run for the report."""
    risk = compute_empirical_risk(records)
    claim_status_counts: Dict[str, int] = {}
    failure_type_counts: Dict[str, int] = {}
    materiality_counts: Dict[str, int] = {}
    for r in records:
        for c in r.claims:
            claim_status_counts[c.claim_status.value] = claim_status_counts.get(c.claim_status.value, 0) + 1
            failure_type_counts[c.failure_type.value] = failure_type_counts.get(c.failure_type.value, 0) + 1
            materiality_counts[c.materiality.value] = materiality_counts.get(c.materiality.value, 0) + 1
    return {
        "n_evaluated": len(records),
        "risk": risk,
        "claim_status_distribution": claim_status_counts,
        "failure_taxonomy_distribution": failure_type_counts,
        "materiality_distribution": materiality_counts,
        "needs_human_review_count": sum(1 for r in records if r.needs_human_review),
    }
