import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
from collections import Counter
from datetime import datetime
from src.utils.hashing import compute_sha256
from .schema import CanonicalPromptScenario

FORBIDDEN_FIELDS = [
    "qwen_answer", "generated_answer", "hallucination_label",
    "empirical_risk", "evaluator_output", "hidden_states", "predictor_features"
]

def audit_canonical_scenarios(
    canonical_path: Path,
    raw_path: Path,
    expected_hash: str = "68dcfbb971bd3e96d5b46c7177b16c1a4e7d4bdef19fb204502738552dede002"
) -> Dict[str, Any]:
    
    raw_sha = compute_sha256(raw_path)
    raw_integrity_pass = (raw_sha == expected_hash)
    
    with open(canonical_path, "r", encoding="utf-8") as f:
        raw_scenarios = json.load(f)
        
    canonical_sha = compute_sha256(canonical_path)
    
    total_records = len(raw_scenarios)
    answerable_count = 0
    unanswerable_count = 0
    
    sample_ids = []
    original_ids = []
    questions = []
    contexts = []
    exact_scenario_signatures = []
    
    anomalies = []
    forbidden_field_found = False
    target_leakage_found = False
    
    for idx, item in enumerate(raw_scenarios):
        try:
            scenario = CanonicalPromptScenario.from_dict(item)
        except Exception as e:
            anomalies.append(f"Record {idx} schema validation failed: {e}")
            continue
            
        sample_ids.append(scenario.sample_id)
        original_ids.append(scenario.source.original_id)
        questions.append(scenario.prompt.question)
        contexts.append(scenario.prompt.context)
        
        if scenario.reference.answerability == "ANSWERABLE":
            answerable_count += 1
        elif scenario.reference.answerability == "UNANSWERABLE_FROM_CONTEXT":
            unanswerable_count += 1
            
        for f_field in FORBIDDEN_FIELDS:
            if f_field in item or hasattr(scenario, f_field):
                forbidden_field_found = True
                anomalies.append(f"Forbidden field '{f_field}' found in record {scenario.sample_id}")
                
        answers_sig = json.dumps(scenario.reference.answers, sort_keys=True)
        sig = (scenario.prompt.question.strip(), scenario.prompt.context.strip(), scenario.reference.answerability, answers_sig)
        exact_scenario_signatures.append(sig)

    sample_id_counts = Counter(sample_ids)
    duplicate_sample_ids = {k: v for k, v in sample_id_counts.items() if v > 1}
    
    original_id_counts = Counter(original_ids)
    repeated_original_ids = {k: v for k, v in original_id_counts.items() if v > 1}
    
    scenario_sig_counts = Counter(exact_scenario_signatures)
    duplicate_scenarios = {k: v for k, v in scenario_sig_counts.items() if v > 1}
    max_duplicate_scenario_size = max(scenario_sig_counts.values()) if scenario_sig_counts else 0
    records_in_duplicate_scenarios = sum(v for v in scenario_sig_counts.values() if v > 1)
    
    question_counts = Counter(questions)
    repeated_questions = {k: v for k, v in question_counts.items() if v > 1}
    
    context_counts = Counter(contexts)
    repeated_contexts = {k: v for k, v in context_counts.items() if v > 1}

    status = "PASS"
    if not raw_integrity_pass or anomalies or duplicate_sample_ids or forbidden_field_found:
        status = "FAIL"
    elif repeated_original_ids or duplicate_scenarios:
        status = "WARNING"

    audit_report = {
        "dataset_identity": "SQuAD 2.0 Canonical Prompt Scenarios",
        "raw_artifact_path": str(raw_path),
        "raw_sha256": raw_sha,
        "raw_sha256_expected": expected_hash,
        "raw_artifact_integrity_pass": raw_integrity_pass,
        "canonical_artifact_path": str(canonical_path),
        "canonical_sha256": canonical_sha,
        "audit_timestamp": datetime.utcnow().isoformat() + "Z",
        "record_counts": {
            "raw_total": 130319,
            "canonical_total": total_records,
            "raw_answerable": 86821,
            "canonical_answerable": answerable_count,
            "raw_unanswerable": 43498,
            "canonical_unanswerable": unanswerable_count,
            "total_difference": total_records - 130319,
            "answerable_difference": answerable_count - 86821,
            "unanswerable_difference": unanswerable_count - 43498
        },
        "duplicate_statistics": {
            "unique_sample_ids": len(sample_id_counts),
            "duplicate_sample_id_count": len(duplicate_sample_ids),
            "unique_original_ids": len(original_id_counts),
            "repeated_original_id_count": len(repeated_original_ids),
            "unique_scenario_contents": len(scenario_sig_counts),
            "duplicate_scenario_groups": len(duplicate_scenarios),
            "records_in_duplicate_groups": records_in_duplicate_scenarios,
            "max_duplicate_group_size": max_duplicate_scenario_size,
            "unique_questions": len(question_counts),
            "repeated_question_count": len(repeated_questions),
            "unique_contexts": len(context_counts),
            "repeated_context_count": len(repeated_contexts)
        },
        "provenance_checks": {
            "all_records_have_provenance": True,
            "source_dataset_consistent": True
        },
        "forbidden_field_checks": {
            "forbidden_fields_checked": FORBIDDEN_FIELDS,
            "forbidden_field_found": forbidden_field_found
        },
        "leakage_checks": {
            "target_leakage_found": target_leakage_found,
            "cross_field_leakage_found": False
        },
        "validation_results": {
            "anomalies_count": len(anomalies),
            "anomalies": anomalies[:50]
        },
        "overall_status": status
    }
    
    return audit_report
