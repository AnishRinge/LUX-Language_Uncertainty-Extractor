"""CP4.2.5 representative validation. Execute only with explicit approval.

The frozen design is six selected SQuAD v2 prompts x two protocols x three
seeds: exactly 36 calls. This module does not alter prior artifacts or data.
"""
import json
import os
import re
import statistics
import sys
import time
from datetime import datetime, timezone

import torch

import run_cp4_2_pilot as cp42
import run_cp4_2_sampling_diagnostic as diag

EXPERIMENT = "CP4.2.5"
SEEDS = (1001, 1002, 1003)
PROMPTS = (
    {"dataset_index": 19919, "answerability": "ANSWERABLE", "class": "answerable"},
    {"dataset_index": 61608, "answerability": "ANSWERABLE", "class": "answerable"},
    {"dataset_index": 108548, "answerability": "ANSWERABLE", "class": "answerable"},
    {"dataset_index": 31073, "answerability": "UNANSWERABLE_FROM_CONTEXT", "class": "unanswerable"},
    {"dataset_index": 69929, "answerability": "UNANSWERABLE_FROM_CONTEXT", "class": "unanswerable"},
    {"dataset_index": 108653, "answerability": "UNANSWERABLE_FROM_CONTEXT", "class": "unanswerable"},
)
PROTOCOLS = (
    {"label": "A", "name": "original CP4.2", "temperature": 0.7, "top_p": 0.8, "top_k": 20},
    {"label": "B", "name": "exploratory T=1.3/top_p=1.0", "temperature": 1.3, "top_p": 1.0, "top_k": 20},
)
TOP_K = 20
MAX_NEW_TOKENS = cp42.MAX_NEW_TOKENS
PLANNED_GENERATION_COUNT = len(PROMPTS) * len(PROTOCOLS) * len(SEEDS)
ARTIFACT_PATH = "data/pilots/cp4_2_5_representative_validation.json"
REPORT_PATH = "docs/cp4_2_5_representative_validation_report.md"
CANONICAL_DATASET_PATH = "data/prompts/squad_v2_canonical_scenarios.json"
CP4_2_ARTIFACT_PATH = cp42.PILOT_ARTIFACT_PATH
CP4_2_ARTIFACT_SHA256_EXPECTED = "10ac27699e42b6c6627f8bd13ad7f1267ed20a4885b4f5fac9f12ece54eb5365"
CROSS_CHECK_TOL = 0.0

CLASSIFICATION_SEVERITY = {
    "identical": 0, "formatting-only": 1,
    "orthographic/tokenization-only": 2,
    "lexical but semantically equivalent": 3,
    "materially different wording": 4, "semantically different answer": 5,
}
NON_SEMANTIC_CLASSES = (
    "identical", "formatting-only", "orthographic/tokenization-only",
    "lexical but semantically equivalent",
)
FRAMING_CLAUSE_TOKENS = 8
MIN_SHARED_SUBSTANTIVE_TOKENS = 10


def _now():
    return datetime.now(timezone.utc).isoformat()


def sha256_of_file(path):
    return diag.sha256_of_file(path)


def _atomic_write_json(path, payload):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def build_processor_chain(temperature, top_p):
    """Mirror the sampling processor order: Temperature -> TopK -> TopP."""
    from transformers import LogitsProcessorList, TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper
    return LogitsProcessorList([
        TemperatureLogitsWarper(temperature),
        TopKLogitsWarper(top_k=TOP_K, min_tokens_to_keep=1),
        TopPLogitsWarper(top_p=top_p, min_tokens_to_keep=1),
    ])


def _ws_norm(text):
    return re.sub(r"\s+", " ", (text or "")).strip()


def _alnum_squash(text):
    return re.sub(r"[^a-z0-9]", "", (text or "").lower())


def _tokens(text):
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def classify_pair(a, b):
    """Never infer semantic equivalence from a material text difference."""
    if a == b:
        return "identical"
    if _ws_norm(a) == _ws_norm(b):
        return "formatting-only"
    if _alnum_squash(a) == _alnum_squash(b):
        return "orthographic/tokenization-only"
    return "materially different wording"


def _manual_review(a, b):
    """Predeclared conservative framing-clause review; no numeric score."""
    toks_a, toks_b = _tokens(a), _tokens(b)
    shared = set(toks_a) & set(toks_b)
    numbers_a = sorted(token for token in toks_a if token.isdigit())
    numbers_b = sorted(token for token in toks_b if token.isdigit())
    positions_a = [i for i, token in enumerate(toks_a) if token not in shared]
    positions_b = [i for i, token in enumerate(toks_b) if token not in shared]
    equivalent = bool(numbers_a == numbers_b and positions_a and positions_b
                      and max(positions_a) < FRAMING_CLAUSE_TOKENS
                      and max(positions_b) < FRAMING_CLAUSE_TOKENS
                      and len(shared) >= MIN_SHARED_SUBSTANTIVE_TOKENS)
    return {
        "numeric_content_identical": numbers_a == numbers_b,
        "shared_substantive_token_count": len(shared),
        "non_shared_token_positions_a": positions_a,
        "non_shared_token_positions_b": positions_b,
        "difference_confined_to_framing_clause": bool(positions_a and positions_b and max(positions_a) < FRAMING_CLAUSE_TOKENS and max(positions_b) < FRAMING_CLAUSE_TOKENS),
        "classification": "lexical but semantically equivalent" if equivalent else "materially different wording",
        "requires_human_semantic_adjudication": not equivalent,
    }


def analyze_semantic_diversity(generations):
    distinct = []
    for generation in generations:
        output = generation["generated_answer"]
        entry = next((item for item in distinct if item["output"] == output), None)
        if entry is None:
            distinct.append({"output": output, "protocols": [generation["protocol"]], "seeds": [generation["seed"]]})
        else:
            if generation["protocol"] not in entry["protocols"]:
                entry["protocols"].append(generation["protocol"])
            entry["seeds"].append(generation["seed"])
    pairs, reviews = [], []
    for i, left in enumerate(distinct):
        for right in distinct[i + 1:]:
            pair = {"output_a": left["output"], "output_b": right["output"], "classification": classify_pair(left["output"], right["output"])}
            if pair["classification"] == "materially different wording":
                review = _manual_review(left["output"], right["output"])
                review.update({"output_a": left["output"], "output_b": right["output"]})
                reviews.append(review)
                pair["classification"] = review["classification"]
            pairs.append(pair)
    worst = max((pair["classification"] for pair in pairs), key=lambda value: CLASSIFICATION_SEVERITY[value], default="identical")
    unresolved = [review for review in reviews if review["requires_human_semantic_adjudication"]]
    return {
        "method": "Pairwise exact/whitespace/alphanumeric classification, then the predeclared conservative framing-clause review; no numeric semantic-diversity score.",
        "distinct_output_count": len(distinct), "distinct_outputs": distinct,
        "pairwise_classifications": pairs, "manual_review": reviews,
        "most_severe_classification": worst, "unresolved_pairs": unresolved,
        "semantic_answer_diversity": bool(unresolved), "semantic_diversity_zero": not unresolved,
        "note": "Unresolved wording differences are potential semantic diversity, never automatically labelled semantically different answers.",
    }


def _stats(generations):
    steps = [step for generation in generations for step in generation["steps"]]
    nuclei = [step["nucleus_size"] for step in steps]
    top1 = [step["top1_prob"] for step in steps]
    ranks = [step["sampled_rank"] for step in steps if step["sampled_rank"] is not None]
    return {
        "generation_count": len(generations), "eos_reached": sum(g["eos_reached"] for g in generations),
        "truncated": sum(g["reached_max_tokens"] for g in generations),
        "unique_final_outputs": len({g["generated_answer"] for g in generations}),
        "total_generation_steps": len(steps),
        "nucleus_size": {"mean": round(statistics.mean(nuclei), 4) if nuclei else None, "min": min(nuclei) if nuclei else None, "max": max(nuclei) if nuclei else None},
        "top1_probability": {"mean": round(statistics.mean(top1), 4) if top1 else None, "min": round(min(top1), 4) if top1 else None, "max": round(max(top1), 4) if top1 else None},
        "sampled_rank_distribution": {str(rank): ranks.count(rank) for rank in sorted(set(ranks))},
    }


def _checkpoint(started, generations, cross_checks, calls, error=None):
    return {"experiment": EXPERIMENT, "run_state": "FAILED" if error else "IN_PROGRESS", "run_started_at": started, "run_updated_at": _now(), "run_error": error, "planned_generation_count": PLANNED_GENERATION_COUNT, "model_generate_calls": calls, "generations": generations, "cross_check": {"tolerance": CROSS_CHECK_TOL, "results": cross_checks}}


def _load_checkpoint():
    if not os.path.exists(ARTIFACT_PATH):
        return None
    with open(ARTIFACT_PATH, encoding="utf-8") as handle:
        artifact = json.load(handle)
    if artifact.get("experiment") != EXPERIMENT:
        raise RuntimeError("existing artifact is not CP4.2.5")
    if artifact.get("run_state") == "COMPLETED":
        raise RuntimeError("completed CP4.2.5 artifact: refusing to rerun generations")
    return artifact


def _prompt_metadata(scenario, prompt):
    source, metadata = scenario.get("source", {}), scenario.get("metadata", {})
    return {"dataset_index": prompt["dataset_index"], "sample_id": scenario.get("sample_id"), "answerability": scenario.get("reference", {}).get("answerability"), "class": prompt["class"], "expected_behavior": metadata.get("expected_behavior"), "original_id": source.get("original_id"), "article_id": source.get("article_id")}


def _fidelity_reference():
    with open(CP4_2_ARTIFACT_PATH, encoding="utf-8") as handle:
        artifact = json.load(handle)
    return next((g for g in artifact["generations"] if g["dataset_index"] == 61608 and g["seed"] == 1001), None)


def _final_artifact(started, load_time, metadata, generations, checks, calls, canonical_sha, cp42_sha):
    by_prompt = {}
    for prompt in PROMPTS:
        group = [g for g in generations if g["dataset_index"] == prompt["dataset_index"]]
        by_prompt[str(prompt["dataset_index"])] = {"metadata": metadata[str(prompt["dataset_index"])], "statistics": _stats(group), "semantic_diversity": analyze_semantic_diversity(group)}
    by_protocol = {}
    for protocol in PROTOCOLS:
        group = [g for g in generations if g["protocol"] == protocol["label"]]
        by_protocol[protocol["label"]] = {"definition": protocol, "statistics": _stats(group), "semantic_diversity": analyze_semantic_diversity(group)}
    answerability = {}
    for group_name in ("answerable", "unanswerable"):
        group = [g for g in generations if g["answerability_class"] == group_name]
        answerability[group_name] = {"statistics": _stats(group), "semantic_diversity": analyze_semantic_diversity(group)}
    anchor = next((g for g in generations if (g["dataset_index"], g["protocol"], g["seed"]) == (61608, "A", 1001)), None)
    reference = _fidelity_reference()
    fidelity = {"anchor": {"dataset_index": 61608, "protocol": "A", "seed": 1001}, "cp4_2_sha256_expected": CP4_2_ARTIFACT_SHA256_EXPECTED, "cp4_2_sha256_computed": cp42_sha, "reference_found": reference is not None, "generation_found": anchor is not None}
    if anchor and reference:
        # CP4.2 predates token-ID persistence.  Do not fabricate a token-ID
        # comparison: compare the fields that the read-only reference exposes.
        fidelity.update({"text_match": anchor["generated_answer"] == reference["generated_answer"], "token_count_match": anchor["generated_token_count"] == reference["generated_token_count"], "token_ids_comparison": "unavailable: CP4.2 pilot artifact has no gen_token_ids field"})
        fidelity["match"] = all(fidelity[key] for key in ("text_match", "token_count_match"))
    return {"experiment": EXPERIMENT, "run_state": "COMPLETED", "run_started_at": started, "run_updated_at": _now(), "run_error": None, "model": {"name": cp42.MODEL_NAME, "revision": cp42.MODEL_REVISION, "device": cp42.DEVICE, "execution_strategy": cp42.EXECUTION_STRATEGY, "frozen": True, "eval": True, "gradients": False, "thinking_enforcement": cp42.THINKING_ENFORCEMENT}, "execution": {"cpu_only": True, "load_time_sec": load_time}, "protocols": list(PROTOCOLS), "seeds": list(SEEDS), "prompt_metadata": metadata, "planned_generation_count": PLANNED_GENERATION_COUNT, "model_generate_calls": calls, "generations": generations, "cross_check": {"tolerance": CROSS_CHECK_TOL, "results": checks, "all_pass": all(item["max_abs_diff"] <= CROSS_CHECK_TOL for item in checks)}, "per_prompt_statistics": by_prompt, "per_protocol_statistics": by_protocol, "answerability_comparison": answerability, "semantic_diversity": analyze_semantic_diversity(generations), "fidelity": fidelity, "artifact_integrity": {"cp4_2_artifact_unchanged": cp42_sha == CP4_2_ARTIFACT_SHA256_EXPECTED, "canonical_dataset_sha256": canonical_sha, "canonical_dataset_unchanged_during_run": canonical_sha == sha256_of_file(CANONICAL_DATASET_PATH)}}


def write_report(artifact):
    """Generate the report exclusively from the completed artifact."""
    lines = ["# CP4.2.5 Representative Validation", "", "## Exact protocol", "", "- Model: `{}` @ `{}`; CPU-only, frozen/eval/no gradients.".format(artifact["model"]["name"], artifact["model"]["revision"]), "- Thinking disabled through `{}`.".format(artifact["model"]["thinking_enforcement"]), "- Prompts: {}. Seeds: {}.".format(", ".join(str(p["dataset_index"]) for p in PROMPTS), ", ".join(str(s) for s in SEEDS)), "- A: temperature=0.7, top_p=0.8, top_k=20; B: temperature=1.3, top_p=1.0, top_k=20.", "", "## Results", "", "- Generation success: `{}` / `{}`; processor/logits cross-check at tolerance 0.0: `{}`.".format(artifact["model_generate_calls"], artifact["planned_generation_count"], artifact["cross_check"]["all_pass"]), "- Fidelity anchor (61608/A/1001): `{}`.".format(artifact["fidelity"].get("match")), "", "| Protocol | Generations | EOS | Truncated | Unique outputs |", "|---|---:|---:|---:|---:|"]
    for label, item in artifact["per_protocol_statistics"].items():
        stats = item["statistics"]
        lines.append("| {} | {} | {} | {} | {} |".format(label, stats["generation_count"], stats["eos_reached"], stats["truncated"], stats["unique_final_outputs"]))
    overall = artifact["semantic_diversity"]
    lines += ["", "## Diversity and scope", "", "The six-class conservative framework was applied per prompt, protocol, answerability class, and overall. Overall most severe class: **{}**. Potential semantic diversity requiring human adjudication: **{}**.".format(overall["most_severe_classification"], overall["semantic_answer_diversity"]), "", "## Limitations and conservative conclusion", "", "This is a fixed six-prompt representative sample with three seeds per protocol. It does not justify dataset-wide conclusions or a production decoding change. No numeric semantic-diversity score is claimed."]
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def run_experiment():
    """Run only when authorized. Tests must never invoke this entry point."""
    if PLANNED_GENERATION_COUNT != 36:
        raise RuntimeError("plan must remain exactly 36 generations")
    cp42_sha = sha256_of_file(CP4_2_ARTIFACT_PATH)
    if cp42_sha != CP4_2_ARTIFACT_SHA256_EXPECTED:
        raise RuntimeError("CP4.2 fidelity artifact SHA256 mismatch")
    prior = _load_checkpoint()
    started = prior.get("run_started_at") if prior else _now()
    generations = prior.get("generations", []) if prior else []
    checks = prior.get("cross_check", {}).get("results", []) if prior else []
    calls = int(prior.get("model_generate_calls", len(generations))) if prior else 0
    completed = {(g["dataset_index"], g["protocol"], g["seed"]) for g in generations}
    if calls != len(generations) or calls > PLANNED_GENERATION_COUNT:
        raise RuntimeError("invalid checkpoint call count")
    canonical_sha = sha256_of_file(CANONICAL_DATASET_PATH)
    scenarios = cp42._load_scenarios()
    scenarios = {prompt["dataset_index"]: scenarios[prompt["dataset_index"]] for prompt in PROMPTS}
    metadata = {str(prompt["dataset_index"]): _prompt_metadata(scenarios[prompt["dataset_index"]], prompt) for prompt in PROMPTS}
    print("Loading model (CP4.2 loader)...", flush=True)
    started_loading = time.time()
    tokenizer, model = cp42._load_model()
    load_time = time.time() - started_loading
    encoded = {}
    for prompt in PROMPTS:
        index = prompt["dataset_index"]
        rendered = diag.render_prompt_for_scenario(tokenizer, scenarios[index])
        if diag._NON_THINKING_MARKER not in rendered:
            raise RuntimeError("Qwen3 chat template did not emit non-thinking prefix")
        encoded[index] = (rendered, tokenizer(rendered, return_tensors="pt").to(cp42.DEVICE))
    for prompt in PROMPTS:
        for protocol in PROTOCOLS:
            chain = build_processor_chain(protocol["temperature"], protocol["top_p"])
            for seed in SEEDS:
                key = (prompt["dataset_index"], protocol["label"], seed)
                if key in completed:
                    continue
                if calls >= PLANNED_GENERATION_COUNT:
                    raise RuntimeError("generation budget exhausted")
                rendered, inputs = encoded[prompt["dataset_index"]]
                torch.manual_seed(seed); begun = time.time()
                with torch.no_grad():
                    output = model.generate(**inputs, do_sample=True, temperature=protocol["temperature"], top_p=protocol["top_p"], top_k=TOP_K, max_new_tokens=MAX_NEW_TOKENS, pad_token_id=(tokenizer.pad_token_id or tokenizer.eos_token_id), output_scores=True, output_logits=True, return_dict_in_generate=True)
                calls += 1
                steps, token_ids, text, count, eos, max_diff = diag.analyze_output(output, tokenizer, int(inputs["input_ids"].shape[1]), chain, inputs["input_ids"])
                checks.append({"dataset_index": prompt["dataset_index"], "protocol": protocol["label"], "seed": seed, "max_abs_diff": float(max_diff)})
                generations.append({"dataset_index": prompt["dataset_index"], "sample_id": scenarios[prompt["dataset_index"]].get("sample_id"), "answerability": prompt["answerability"], "answerability_class": prompt["class"], "protocol": protocol["label"], "seed": seed, "decoding_config": {"do_sample": True, "temperature": protocol["temperature"], "top_p": protocol["top_p"], "top_k": TOP_K, "max_new_tokens": MAX_NEW_TOKENS}, "prompt_text": rendered, "prompt_token_count": int(inputs["input_ids"].shape[1]), "generated_answer": text, "gen_token_ids": [int(value) for value in token_ids], "generated_token_count": int(count), "eos_reached": bool(eos), "reached_max_tokens": bool(count >= MAX_NEW_TOKENS), "steps": steps, "cross_check_max_abs_diff": float(max_diff), "latency_sec": float(time.time() - begun), "status": "SUCCESS"})
                completed.add(key)
                if max_diff > CROSS_CHECK_TOL:
                    error = "processor/logits cross-check failed; no further generations performed"
                    _atomic_write_json(ARTIFACT_PATH, _checkpoint(started, generations, checks, calls, error))
                    raise RuntimeError(error)
                _atomic_write_json(ARTIFACT_PATH, _checkpoint(started, generations, checks, calls))
    if calls != PLANNED_GENERATION_COUNT or len(generations) != PLANNED_GENERATION_COUNT:
        raise RuntimeError("experiment did not complete exactly 36 generations")
    artifact = _final_artifact(started, load_time, metadata, generations, checks, calls, canonical_sha, sha256_of_file(CP4_2_ARTIFACT_PATH))
    _atomic_write_json(ARTIFACT_PATH, artifact)
    write_report(artifact)
    return artifact


def main():
    run_experiment()
    return 0


if __name__ == "__main__":
    sys.exit(main())
