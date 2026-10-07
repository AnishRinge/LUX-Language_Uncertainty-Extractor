"""CP4.2 - Generation protocol pilot for the LUX target LLM.

Scope: generation-protocol infrastructure only. This module deliberately does NOT
implement hallucination evaluation, empirical risk, hidden-state extraction, or
predictor features.

All protocol logic (user-message construction, chat-template rendering,
deterministic pilot selection, generation-record metadata, diversity metrics,
verdict evaluation, incremental artifact persistence, report rendering) is
implemented as importable pure functions using only the standard library, so it
can be unit tested without loading Qwen3 weights. ``torch``, ``transformers``
and ``psutil`` are imported lazily inside the execution path only.

Locked protocol (CP4.2):
    model             Qwen/Qwen3-1.7B @ 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e
    device            CPU_ONLY (no CUDA torch build is permitted)
    thinking          DISABLED at runtime via the Qwen3 chat template
    do_sample         True
    temperature       0.7
    top_p             0.8
    top_k             20
    max_new_tokens    512
    generations/prompt 10, differing ONLY by random seed
"""

import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

# --------------------------------------------------------------------------
# Locked protocol constants
# --------------------------------------------------------------------------

MODEL_NAME = "Qwen/Qwen3-1.7B"
MODEL_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
DEVICE = "cpu"
EXECUTION_STRATEGY = "CPU_ONLY"

# FIX 1: thinking is disabled through the Qwen3 chat template. The template
# branch `{%- if enable_thinking is defined and enable_thinking is false %}`
# appends an empty `<think>\n\n</think>\n\n` block to the assistant generation
# prefix, which is Qwen3's supported mechanism for non-thinking generation.
THINKING_MODE = False
THINKING_ENFORCEMENT = "qwen3_chat_template:enable_thinking=False"
PROMPT_FORMAT = "qwen3_chat_template"
SYSTEM_PROMPT = (
    "You are a precise question answering assistant. Answer using only the "
    "context provided. If the context does not contain the answer, say that you "
    "do not know. Be concise."
)

DO_SAMPLE = True
TEMPERATURE = 0.7
TOP_P = 0.8
TOP_K = 20
MAX_NEW_TOKENS = 512

SEEDS = (1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010)
GENERATIONS_PER_PROMPT = len(SEEDS)

# FIX 3: small deterministic stratified pilot (3 answerable + 3 unanswerable).
ANSWERABLE_PILOT_SIZE = 3
UNANSWERABLE_PILOT_SIZE = 3
PILOT_SIZE = ANSWERABLE_PILOT_SIZE + UNANSWERABLE_PILOT_SIZE
PILOT_SELECTION_RULE = (
    "Deterministic mid-point stratification over the canonical dataset in file "
    "order. Positions of each answerability class are collected first, then "
    "evenly_spaced_indices(len(positions), k) selects k mid-point indices. No "
    "random sampling is used, so selection is reproducible for a given dataset. "
    "Class is read from reference.answerability "
    "(ANSWERABLE / UNANSWERABLE_FROM_CONTEXT)."
)

CANONICAL_SCENARIOS_PATH = Path("data/prompts/squad_v2_canonical_scenarios.json")
PILOT_ARTIFACT_PATH = Path("data/pilots/cp4_2_generation_pilot.json")
PILOT_REPORT_PATH = Path("docs/cp4_2_generation_protocol_pilot_report.md")

ANSWERABLE = "ANSWERABLE"
UNANSWERABLE = "UNANSWERABLE_FROM_CONTEXT"

STATUS_SUCCESS = "SUCCESS"
STATUS_FAILED = "FAILED"
RUN_IN_PROGRESS = "IN_PROGRESS"
RUN_COMPLETED = "COMPLETED"
RUN_FAILED = "FAILED"

VERDICT_PASS = "PASS"
VERDICT_FAIL = "FAIL"
VERDICT_NEEDS_REVISION = "NEEDS_REVISION"

# FIX 9: documented, non-arbitrary acceptance criteria.
# With do_sample=True / temperature=0.7 / top_p=0.8 / top_k=20, ten samples of
# the same prompt from a 1.7B model should not collapse to a single string.
# Collapsing to one unique output across ten seeds means sampling is
# effectively deterministic and the protocol is not demonstrating the
# stochastic behaviour the risk target depends on.
MIN_UNIQUE_OUTPUTS_PER_PROMPT = 2
# If most generations hit the 512-token cap the protocol is not eliciting
# terminated answers, which weakens generation-level evaluation downstream.
MAX_ACCEPTABLE_TRUNCATION_RATE = 0.5

# FIX 10: fields that must never appear in a CP4.2 record.
# `generated_answer` is intentionally NOT here: generation records are the
# legitimate owner of that field. The canonical prompt dataset must remain
# free of it, which is enforced separately by src/dataset/schema.py.
FORBIDDEN_RECORD_FIELDS = (
    "hallucination_label",
    "response_label",
    "empirical_risk",
    "empirical_hallucination_risk",
    "evaluator_output",
    "predictor_features",
    "hidden_states",
    "hidden_state",
    "ppl",
    "perplexity",
)

REQUIRED_RECORD_FIELDS = (
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
)


def generation_config() -> dict:
    """The frozen decoding configuration recorded with every pilot result."""
    return {
        "model_name": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "device": DEVICE,
        "execution_strategy": EXECUTION_STRATEGY,
        "thinking_mode": THINKING_MODE,
        "thinking_enforcement": THINKING_ENFORCEMENT,
        "prompt_format": PROMPT_FORMAT,
        "do_sample": DO_SAMPLE,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "top_k": TOP_K,
        "max_new_tokens": MAX_NEW_TOKENS,
        "generations_per_prompt": GENERATIONS_PER_PROMPT,
        "seed_list": list(SEEDS),
    }


# --------------------------------------------------------------------------
# FIX 1 / FIX 2 - prompt construction
# --------------------------------------------------------------------------

def build_user_message(context: str, question: str) -> str:
    """Build the single user message for a scenario.

    Only the SQuAD context and question are carried over verbatim. The reference
    answer, the answerability label and every other scenario field are
    deliberately excluded so that nothing beyond the intended prompt/context
    reaches the model. The instruction line is a fixed protocol instruction
    (identical for every scenario), not scenario content, and is required for
    abstention to be observable on UNANSWERABLE_FROM_CONTEXT items.
    """
    return "{context}\n\n{question}".format(context=context, question=question)


def build_chat_messages(context: str, question: str) -> list:
    """System + user messages for Qwen3's chat template."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_message(context, question)},
    ]


def render_prompt(tokenizer, context: str, question: str) -> str:
    """Render the Qwen3 chat prompt with thinking disabled.

    `enable_thinking=False` is the tokenizer's documented switch. Verified
    against the locally cached Qwen3-1.7B tokenizer: the rendered assistant
    generation prefix becomes `<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n`,
    whereas omitting the argument yields a bare `<|im_start|>assistant\\n`
    prefix that leaves the model free to emit reasoning. Omitting the argument
    therefore changes real runtime behaviour, which is why the flag is passed
    explicitly rather than only recorded as metadata.
    """
    return tokenizer.apply_chat_template(
        build_chat_messages(context, question),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


# --------------------------------------------------------------------------
# FIX 3 - deterministic stratified pilot selection
# --------------------------------------------------------------------------

def evenly_spaced_indices(count: int, k: int) -> list:
    """k deterministic, strictly increasing, mid-point indices into range(count).

    Deterministic and free of randomness. Spreading across mid-points avoids
    picking several near-identical records from the same SQuAD article, which a
    head-of-list slice would do because the canonical dataset is article-ordered.
    """
    if count <= 0 or k <= 0:
        return []
    if count <= k:
        return list(range(count))
    step = count / k
    return [min(count - 1, int((i + 0.5) * step)) for i in range(k)]


def is_answerable(scenario: dict) -> bool:
    return scenario.get("reference", {}).get("answerability") == ANSWERABLE


def select_pilot_scenarios(
    scenarios,
    n_answerable: int = ANSWERABLE_PILOT_SIZE,
    n_unanswerable: int = UNANSWERABLE_PILOT_SIZE,
) -> list:
    """Select a small deterministic stratified pilot.

    Returns [(dataset_index, scenario), ...] with n_answerable answerable and
    n_unanswerable unanswerable scenarios. Deterministic: the same input list
    always yields the same output, with no random sampling.
    """
    answerable_positions = []
    unanswerable_positions = []
    for index, scenario in enumerate(scenarios):
        if is_answerable(scenario):
            answerable_positions.append(index)
        else:
            unanswerable_positions.append(index)

    selected = []
    for positions, wanted in (
        (answerable_positions, n_answerable),
        (unanswerable_positions, n_unanswerable),
    ):
        for slot in evenly_spaced_indices(len(positions), wanted):
            dataset_index = positions[slot]
            selected.append((dataset_index, scenarios[dataset_index]))
    return selected


def describe_selection(selected) -> list:
    """Provenance for each selected scenario (never used as predictor input)."""
    described = []
    for dataset_index, scenario in selected:
        source = scenario.get("source", {})
        metadata = scenario.get("metadata", {})
        described.append(
            {
                "dataset_index": dataset_index,
                "sample_id": scenario.get("sample_id"),
                "answerability": scenario.get("reference", {}).get("answerability"),
                "expected_behavior": metadata.get("expected_behavior"),
                "phenomenon": metadata.get("phenomenon"),
                "article_id": source.get("article_id"),
                "original_id": source.get("original_id"),
                "note": "scenario provenance for auditability; never a predictor feature",
            }
        )
    return described


# --------------------------------------------------------------------------
# FIX 10 - generation record metadata
# --------------------------------------------------------------------------

def format_error(exc: BaseException) -> str:
    return "{0}: {1}".format(type(exc).__name__, exc)


def build_generation_record(
    *,
    scenario,
    dataset_index: int,
    generation_index: int,
    seed: int,
    prompt_text: str,
    prompt_token_count: int,
    status: str,
    generated_answer: str = "",
    generated_token_count: int = 0,
    latency: float = 0.0,
    reached_max_tokens: bool = False,
    error=None,
) -> dict:
    """Build one generation record with the full frozen protocol metadata.

    Seed is recorded as metadata only. No hallucination label, empirical risk,
    evaluator output, predictor feature or hidden state is ever attached here.
    """
    source = scenario.get("source", {})
    metadata = scenario.get("metadata", {})
    return {
        "sample_id": scenario.get("sample_id"),
        "dataset_index": dataset_index,
        "generation_index": generation_index,
        "seed": seed,
        "model_name": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "tokenizer_revision": MODEL_REVISION,
        "device": DEVICE,
        "execution_strategy": EXECUTION_STRATEGY,
        "thinking_mode": THINKING_MODE,
        "thinking_enforcement": THINKING_ENFORCEMENT,
        "prompt_format": PROMPT_FORMAT,
        "do_sample": DO_SAMPLE,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "top_k": TOP_K,
        "max_new_tokens": MAX_NEW_TOKENS,
        "status": status,
        "generated_answer": generated_answer,
        "generated_token_count": generated_token_count,
        "latency": latency,
        "reached_max_tokens": reached_max_tokens,
        "error": error,
        "prompt_token_count": prompt_token_count,
        "prompt_text": prompt_text,
        "provenance": {
            "dataset": source.get("dataset"),
            "split": source.get("split"),
            "original_id": source.get("original_id"),
            "article_id": source.get("article_id"),
            "paragraph_id": source.get("paragraph_id"),
            "article_title": source.get("article_title"),
            "answerability": scenario.get("reference", {}).get("answerability"),
            "expected_behavior": metadata.get("expected_behavior"),
            "note": "scenario provenance for auditability; never a predictor feature",
        },
    }


def find_forbidden_fields(records) -> list:
    """Return every forbidden field found anywhere in the given records."""
    violations = []
    for record in records:
        for field_name in FORBIDDEN_RECORD_FIELDS:
            if field_name in record:
                violations.append({"field": field_name, "keys": sorted(record.keys())})
        provenance = record.get("provenance") or {}
        for field_name in FORBIDDEN_RECORD_FIELDS:
            if field_name in provenance:
                violations.append(
                    {"field": "provenance." + field_name, "keys": sorted(provenance.keys())}
                )
    return violations


def find_missing_fields(records) -> list:
    """Return the required fields missing from any record."""
    missing = []
    for record in records:
        absent = [name for name in REQUIRED_RECORD_FIELDS if name not in record]
        if absent:
            missing.append({"sample_id": record.get("sample_id"), "absent": absent})
    return missing


# --------------------------------------------------------------------------
# FIX 6 - diversity metrics
# --------------------------------------------------------------------------

def normalize_output(text: str) -> str:
    """Comparison key for duplicate detection: trim and collapse whitespace.

    Identical outputs are a legitimate sampling outcome and are never counted
    as failures; they only reduce the measured unique-output ratio.
    """
    return re.sub(r"\s+", " ", (text or "")).strip()


def diversity_metrics(records) -> dict:
    """Diversity metrics for one prompt's generations.

    successful_generations counts only status == SUCCESS. Identical outputs are
    duplicates, not failures.
    """
    successful = [r for r in records if r.get("status") == STATUS_SUCCESS]
    outputs = [r.get("generated_answer", "") for r in successful]
    keys = [normalize_output(o) for o in outputs]
    unique = len(set(keys))
    total = len(successful)
    return {
        "attempted_generations": len(records),
        "successful_generations": total,
        "failed_generations": len(records) - total,
        "unique_outputs": unique,
        "duplicate_outputs": total - unique,
        "unique_output_ratio": (unique / total) if total else 0.0,
    }


def summarize_diversity(prompt_metrics) -> dict:
    """Overall diversity across the whole pilot."""
    attempted = sum(m["attempted_generations"] for m in prompt_metrics)
    successful = sum(m["successful_generations"] for m in prompt_metrics)
    failed = sum(m["failed_generations"] for m in prompt_metrics)
    unique = sum(m["unique_outputs"] for m in prompt_metrics)
    return {
        "num_prompts": len(prompt_metrics),
        "attempted_generations": attempted,
        "successful_generations": successful,
        "failed_generations": failed,
        "unique_outputs": unique,
        "duplicate_outputs": successful - unique,
        "unique_output_ratio": (unique / successful) if successful else 0.0,
        "prompts_with_stochastic_diversity": sum(
            1 for m in prompt_metrics if m["unique_outputs"] >= MIN_UNIQUE_OUTPUTS_PER_PROMPT
        ),
    }


def truncation_rate(records) -> float:
    """Fraction of successful generations that hit the max_new_tokens cap."""
    successful = [r for r in records if r.get("status") == STATUS_SUCCESS]
    if not successful:
        return 0.0
    hit = sum(1 for r in successful if r.get("reached_max_tokens"))
    return hit / len(successful)


# --------------------------------------------------------------------------
# FIX 9 - data-driven verdict
# --------------------------------------------------------------------------

def _check(name, passed, severity, detail):
    return {"name": name, "passed": bool(passed), "severity": severity, "detail": detail}


def evaluate_verdict(
    *,
    run_state: str,
    generations,
    diversity_by_prompt,
    diversity_overall,
    reproducibility,
    artifact_persisted: bool,
    max_truncation_rate: float = MAX_ACCEPTABLE_TRUNCATION_RATE,
    min_unique_outputs_per_prompt: int = MIN_UNIQUE_OUTPUTS_PER_PROMPT,
) -> dict:
    """Derive the CP4.2 verdict from observed results only.

    Required checks that fail force FAIL. Advisory checks that fail force
    NEEDS_REVISION. There is no path that yields PASS merely because the run
    reached the end.
    """
    checks = []
    attempted = len(generations)
    successful = sum(1 for r in generations if r.get("status") == STATUS_SUCCESS)
    failed = attempted - successful

    checks.append(
        _check(
            "pilot_run_completed",
            run_state == RUN_COMPLETED,
            "required",
            "run_state={0}".format(run_state),
        )
    )
    checks.append(
        _check(
            "generations_were_attempted",
            attempted > 0,
            "required",
            "{0} generation(s) recorded".format(attempted),
        )
    )
    checks.append(
        _check(
            "generation_success",
            failed == 0 and successful == attempted,
            "required",
            "{0}/{1} successful, {2} failed".format(successful, attempted, failed),
        )
    )

    reproducibility_ok = bool(reproducibility) and all(
        r.get("status") == STATUS_SUCCESS and r.get("match") is True for r in reproducibility
    )
    reproducibility_detail = "no reproducibility check recorded"
    if reproducibility:
        passed_count = sum(
            1 for r in reproducibility if r.get("status") == STATUS_SUCCESS and r.get("match") is True
        )
        technical_errors = sum(1 for r in reproducibility if r.get("status") != STATUS_SUCCESS)
        reproducibility_detail = "{0}/{1} matched; {2} technical error(s)".format(
            passed_count, len(reproducibility), technical_errors
        )
    checks.append(
        _check("same_seed_reproducibility", reproducibility_ok, "required", reproducibility_detail)
    )

    violations = find_forbidden_fields(generations) + find_forbidden_fields(reproducibility)
    checks.append(
        _check(
            "no_forbidden_fields",
            not violations,
            "required",
            "clean" if not violations else "{0} violation(s)".format(len(violations)),
        )
    )

    missing = find_missing_fields(generations)
    checks.append(
        _check(
            "record_metadata_complete",
            not missing,
            "required",
            "complete" if not missing else "{0} record(s) incomplete".format(len(missing)),
        )
    )

    checks.append(
        _check(
            "artifact_persisted",
            bool(artifact_persisted),
            "required",
            "pilot artifact re-read and validated" if artifact_persisted else "artifact missing or unreadable",
        )
    )

    low_diversity = [
        m for m in diversity_by_prompt if m["unique_outputs"] < min_unique_outputs_per_prompt
    ]
    checks.append(
        _check(
            "stochastic_diversity_observed",
            not low_diversity,
            "advisory",
            "every prompt produced >= {0} unique output(s)".format(min_unique_outputs_per_prompt)
            if not low_diversity
            else "{0}/{1} prompt(s) collapsed to < {1} unique output(s)".format(
                len(low_diversity), len(diversity_by_prompt) or min_unique_outputs_per_prompt
            ),
        )
    )

    rate = truncation_rate(generations)
    checks.append(
        _check(
            "truncation_within_tolerance",
            rate <= max_truncation_rate,
            "advisory",
            "truncation rate {0:.2%} (tolerance {1:.2%})".format(rate, max_truncation_rate),
        )
    )

    required_failed = [c["name"] for c in checks if c["severity"] == "required" and not c["passed"]]
    advisory_failed = [c["name"] for c in checks if c["severity"] == "advisory" and not c["passed"]]

    if required_failed:
        verdict = VERDICT_FAIL
    elif advisory_failed:
        verdict = VERDICT_NEEDS_REVISION
    else:
        verdict = VERDICT_PASS

    return {
        "verdict": verdict,
        "required_failed": required_failed,
        "advisory_failed": advisory_failed,
        "checks": checks,
        "criteria_documentation": {
            "required": (
                "run completed; generations attempted; 100% generation success; same-seed "
                "reproducibility for every checked prompt; no forbidden field in any record; "
                "all required metadata present; pilot artifact re-read successfully."
            ),
            "advisory": (
                "each prompt must yield >= {0} unique outputs across {1} seeds (sampling must "
                "not be effectively deterministic); truncation rate must stay <= {2:.0%} so that "
                "answers terminate rather than always hitting the {3}-token cap."
            ).format(
                min_unique_outputs_per_prompt, GENERATIONS_PER_PROMPT, MAX_ACCEPTABLE_TRUNCATION_RATE, MAX_NEW_TOKENS
            ),
        },
    }


# --------------------------------------------------------------------------
# FIX 7 - incremental artifact persistence
# --------------------------------------------------------------------------

def write_artifact(path, payload) -> None:
    """Atomically (re)write the pilot artifact so it is always valid JSON.

    Writes to a sibling .tmp file, fsyncs, then os.replace()s it into place, so
    a reader never observes a half-written file and a crash mid-write cannot
    destroy the previous checkpoint. Called after every single generation, which
    means an interrupted run still leaves every completed generation on disk.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def verify_artifact(path, expected_generation_count) -> bool:
    """Re-read the artifact and confirm it is valid JSON with the expected rows."""
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return False
    generations = data.get("generations")
    return isinstance(generations, list) and len(generations) == expected_generation_count


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# --------------------------------------------------------------------------
# FIX 9 - report rendering (verdict is data-driven)
# --------------------------------------------------------------------------

def _fmt(value) -> str:
    if isinstance(value, float):
        return "{0:.4f}".format(value)
    return str(value)


def render_report(payload: dict) -> str:
    """Render the CP4.2 report from observed results. No hardcoded verdict."""
    meta = payload["pilot_metadata"]
    verdict_block = payload["verdict"]
    overall = payload["diversity_overall"]
    successes = [g for g in payload["generations"] if g["status"] == STATUS_SUCCESS]

    lines = []
    lines.append("# CP4.2 - Generation Protocol Pilot Report")
    lines.append("")
    lines.append("**Run State:** `{0}`".format(meta["run_state"]))
    lines.append("**Verdict:** **{0}**".format(verdict_block["verdict"]))
    lines.append("**Started:** `{0}`  ".format(meta["run_started_at"]))
    lines.append("**Updated:** `{0}`  ".format(meta["run_updated_at"]))
    lines.append("**Target Model:** `{0}`  ".format(MODEL_NAME))
    lines.append("**Model Revision:** `{0}`  ".format(MODEL_REVISION))
    lines.append("**Execution Strategy:** `{0}`  ".format(EXECUTION_STRATEGY))
    lines.append("")

    if meta.get("run_error"):
        lines.append("## 0. Run Error")
        lines.append("")
        lines.append("```")
        lines.append(str(meta["run_error"]))
        lines.append("```")
        lines.append("")

    lines.append("## 1. Verdict Criteria")
    lines.append("")
    lines.append("The verdict is derived from the checks below. Required failures force **FAIL**;")
    lines.append("advisory failures force **NEEDS_REVISION**.")
    lines.append("")
    lines.append("| Check | Severity | Result | Detail |")
    lines.append("|---|---|---|---|")
    for check in verdict_block["checks"]:
        result = "PASS" if check["passed"] else "FAIL"
        lines.append(
            "| `{0}` | {1} | **{2}** | {3} |".format(
                check["name"], check["severity"], result, check["detail"]
            )
        )
    lines.append("")

    lines.append("## 2. Protocol Executed")
    lines.append("")
    lines.append("- Thinking mode: `{0}` enforced via `{1}`".format(THINKING_MODE, THINKING_ENFORCEMENT))
    lines.append("- Prompt format: `{0}` (chat template, not raw completion)".format(PROMPT_FORMAT))
    lines.append("- `do_sample={0}`, `temperature={1}`, `top_p={2}`, `top_k={3}`".format(
        DO_SAMPLE, TEMPERATURE, TOP_P, TOP_K))
    lines.append("- `max_new_tokens={0}`, generations per prompt = `{1}`".format(
        MAX_NEW_TOKENS, GENERATIONS_PER_PROMPT))
    lines.append("- Seed list: `{0}` (the only variable changed between generations of a prompt)".format(
        list(SEEDS)))
    lines.append("- Selection rule: {0}".format(PILOT_SELECTION_RULE))
    lines.append("")

    lines.append("## 3. Pilot Selection")
    lines.append("")
    lines.append("| # | dataset_index | sample_id | answerability | article_id |")
    lines.append("|---|---|---|---|---|")
    for position, item in enumerate(payload["selection"], start=1):
        lines.append(
            "| {0} | {1} | `{2}` | {3} | {4} |".format(
                position,
                item["dataset_index"],
                item["sample_id"],
                item["answerability"],
                item["article_id"],
            )
        )
    lines.append("")

    lines.append("## 4. Generation Summary")
    lines.append("")
    lines.append("- Prompts planned: `{0}`".format(meta["num_prompts"]))
    lines.append("- Prompts completed: `{0}`".format(meta["prompts_completed"]))
    lines.append("- Generations planned: `{0}`".format(meta["generations_planned"]))
    lines.append("- Generations attempted: `{0}`".format(meta["generations_attempted"]))
    lines.append("- Successful: `{0}`".format(meta["successful_generations"]))
    lines.append("- Failed: `{0}`".format(meta["failed_generations"]))
    if successes:
        lines.append("- Average latency: `{0:.2f}s`".format(
            sum(g["latency"] for g in successes) / len(successes)))
        lines.append("- Average token count: `{0:.1f}`".format(
            sum(g["generated_token_count"] for g in successes) / len(successes)))
    else:
        lines.append("- Average latency: `n/a` (no successful generations)")
        lines.append("- Average token count: `n/a` (no successful generations)")
    lines.append("")

    lines.append("## 5. Diversity Metrics")
    lines.append("")
    lines.append("| sample_id | answerability | success | unique | duplicates | unique ratio |")
    lines.append("|---|---|---|---|---|---|")
    for item in payload["diversity_by_prompt"]:
        lines.append(
            "| `{0}` | {1} | {2} | {3} | {4} | {5:.2%} |".format(
                item["sample_id"],
                item["answerability"],
                item["successful_generations"],
                item["unique_outputs"],
                item["duplicate_outputs"],
                item["unique_output_ratio"],
            )
        )
    lines.append("")
    lines.append("**Overall:** {0}/{1} successful, {2} unique outputs, {3} duplicates, "
                 "unique-output ratio **{4:.2%}**, prompts showing stochastic diversity "
                 "**{5}/{6}**.".format(
                     overall["successful_generations"], overall["attempted_generations"],
                     overall["unique_outputs"], overall["duplicate_outputs"],
                     overall["unique_output_ratio"],
                     overall["prompts_with_stochastic_diversity"], overall["num_prompts"]))
    lines.append("")

    lines.append("## 6. Same-Seed Reproducibility")
    lines.append("")
    if payload["reproducibility"]:
        lines.append("| sample_id | seed | status | match |")
        lines.append("|---|---|---|---|")
        for item in payload["reproducibility"]:
            lines.append("| `{0}` | {1} | {2} | {3} |".format(
                item["sample_id"], item["seed"], item["status"], _fmt(item["match"])))
    else:
        lines.append("No reproducibility check recorded.")
    lines.append("")
    lines.append("Reproducibility generations are stored separately and are excluded from the")
    lines.append("primary generation set.")
    lines.append("")

    failures = [g for g in payload["generations"] if g["status"] != STATUS_SUCCESS]
    lines.append("## 7. Failures")
    lines.append("")
    if failures:
        lines.append("| sample_id | generation_index | seed | error |")
        lines.append("|---|---|---|---|")
        for item in failures:
            lines.append("| `{0}` | {1} | {2} | {3} |".format(
                item["sample_id"], item["generation_index"], item["seed"], item["error"]))
    else:
        lines.append("No generation failed.")
    lines.append("")

    lines.append("## 8. Data Leakage Check")
    lines.append("")
    lines.append("Forbidden fields searched in every generation and reproducibility record:")
    lines.append("`{0}`.".format("`, `".join(FORBIDDEN_RECORD_FIELDS)))
    lines.append("")
    lines.append("- Forbidden field violations: `{0}`".format(len(find_forbidden_fields(payload["generations"]))))
    lines.append("- Artifact re-read validated: `{0}`".format(meta["artifact_persisted"]))
    lines.append("- No hallucination label, empirical risk, evaluator output, predictor feature or")
    lines.append("  hidden state is produced by this checkpoint. Seeds are metadata only.")
    lines.append("")

    lines.append("## 9. Recommendation")
    lines.append("")
    if verdict_block["verdict"] == VERDICT_PASS:
        lines.append("All required and advisory criteria passed on observed results. The protocol")
        lines.append("may be frozen for the next checkpoint.")
    elif verdict_block["verdict"] == VERDICT_NEEDS_REVISION:
        lines.append("Required criteria passed but advisory criteria did not. Review before freezing:")
        lines.append("")
        for name in verdict_block["advisory_failed"]:
            lines.append("- `{0}`".format(name))
    else:
        lines.append("Required criteria failed. The protocol is not validated. Failing criteria:")
        lines.append("")
        for name in verdict_block["required_failed"]:
            lines.append("- `{0}`".format(name))
    lines.append("")
    lines.append("CP4.2 produces generation-protocol infrastructure only. No hallucination")
    lines.append("evaluation, empirical risk, hidden-state extraction or predictor training is")
    lines.append("performed here.")
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Execution (heavy imports are lazy so the module stays cheap to import)
# --------------------------------------------------------------------------

def _load_scenarios():
    with open(CANONICAL_SCENARIOS_PATH, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_model():
    """Load tokenizer + frozen Qwen3 on CPU. Imported lazily on purpose."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, revision=MODEL_REVISION, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        revision=MODEL_REVISION,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.to(DEVICE)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return tokenizer, model


def run_generation_pilot() -> int:
    """Execute the CP4.2 pilot. Returns 0 / 1 / 2 for PASS / FAIL / NEEDS_REVISION.

    Every completed generation is persisted before the next one starts, so an
    interruption, crash or OOM leaves the evidence collected so far intact.
    """
    import os as _os

    import psutil
    import torch

    started = _now()
    state = {
        "pilot_metadata": {
            "checkpoint": "CP4.2",
            "run_state": RUN_IN_PROGRESS,
            "run_started_at": started,
            "run_updated_at": started,
            "run_error": None,
            "generation_config": generation_config(),
            "num_prompts": PILOT_SIZE,
            "prompts_completed": 0,
            "generations_planned": PILOT_SIZE * GENERATIONS_PER_PROMPT,
            "generations_attempted": 0,
            "total_generations": 0,
            "successful_generations": 0,
            "failed_generations": 0,
            "pilot_selection_rule": PILOT_SELECTION_RULE,
            "ram_before_mb": None,
            "ram_after_mb": None,
            "load_time_sec": None,
            "artifact_persisted": False,
        },
        "selection": [],
        "diversity_by_prompt": [],
        "diversity_overall": summarize_diversity([]),
        "verdict": {
            "verdict": VERDICT_NEEDS_REVISION,
            "required_failed": ["pilot_run_completed"],
            "advisory_failed": [],
            "checks": [],
            "criteria_documentation": {},
        },
        "reproducibility": [],
        "generations": [],
    }

    def flush(artifact_persisted=None):
        meta = state["pilot_metadata"]
        meta["run_updated_at"] = _now()
        if artifact_persisted is not None:
            meta["artifact_persisted"] = artifact_persisted
        write_artifact(PILOT_ARTIFACT_PATH, state)

    def log(message):
        print("[{0}] {1}".format(_now(), message), flush=True)

    # FIX 8: model-load failure must still persist evidence.
    try:
        scenarios = _load_scenarios()
        selected = select_pilot_scenarios(scenarios)
        state["selection"] = describe_selection(selected)
        state["pilot_metadata"]["num_prompts"] = len(selected)
        state["pilot_metadata"]["generations_planned"] = len(selected) * GENERATIONS_PER_PROMPT
        log("Selected {0} prompts ({1} answerable / {2} unanswerable) from {3} scenarios".format(
            len(selected),
            sum(1 for _, s in selected if is_answerable(s)),
            sum(1 for _, s in selected if not is_answerable(s)),
            len(scenarios),
        ))

        process = psutil.Process(_os.getpid())
        state["pilot_metadata"]["ram_before_mb"] = process.memory_info().rss / (1024 ** 2)

        log("Loading {0} @ {1} on {2} ...".format(MODEL_NAME, MODEL_REVISION, DEVICE))
        load_start = time.time()
        tokenizer, model = _load_model()
        state["pilot_metadata"]["load_time_sec"] = time.time() - load_start
        state["pilot_metadata"]["ram_after_mb"] = process.memory_info().rss / (1024 ** 2)
        log("Model loaded in {0:.2f}s; frozen={1}; eval={2}".format(
            state["pilot_metadata"]["load_time_sec"],
            not any(p.requires_grad for p in model.parameters()),
            not model.training,
        ))
    except BaseException as exc:  # noqa: BLE001 - evidence must survive any failure
        state["pilot_metadata"]["run_state"] = RUN_FAILED
        state["pilot_metadata"]["run_error"] = format_error(exc)
        flush(artifact_persisted=False)
        log("SETUP FAILED: {0}".format(state["pilot_metadata"]["run_error"]))
        return 1

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    for position, (dataset_index, scenario) in enumerate(selected, start=1):
        sample_id = scenario["sample_id"]
        answerability = scenario["reference"]["answerability"]
        prompt_records = []

        try:
            prompt_text = render_prompt(
                tokenizer, scenario["prompt"]["context"], scenario["prompt"]["question"]
            )
            if "<think>" not in prompt_text:
                raise RuntimeError("chat template did not emit the non-thinking prefix")
            encoded = tokenizer(prompt_text, return_tensors="pt").to(DEVICE)
            prompt_token_count = int(encoded["input_ids"].shape[1])
        except BaseException as exc:  # noqa: BLE001
            log("Prompt {0}/{1} [{2}] FAILED to render: {3}".format(
                position, len(selected), sample_id, format_error(exc)))
            state["pilot_metadata"]["run_error"] = format_error(exc)
            continue

        log("Prompt {0}/{1} [{2}] {3} | {4} prompt tokens | seeds {5}-{6}".format(
            position, len(selected), sample_id, answerability,
            prompt_token_count, SEEDS[0], SEEDS[-1]))

        for generation_index, seed in enumerate(SEEDS, start=1):
            # The ONLY thing that changes between these generations is the seed.
            torch.manual_seed(seed)
            gen_start = time.time()
            status = STATUS_SUCCESS
            answer = ""
            token_count = 0
            error = None
            try:
                with torch.no_grad():
                    outputs = model.generate(
                        **encoded,
                        do_sample=DO_SAMPLE,
                        temperature=TEMPERATURE,
                        top_p=TOP_P,
                        top_k=TOP_K,
                        max_new_tokens=MAX_NEW_TOKENS,
                        pad_token_id=pad_token_id,
                    )
                token_count = int(outputs[0].shape[-1]) - prompt_token_count
                answer = tokenizer.decode(
                    outputs[0][prompt_token_count:], skip_special_tokens=True
                ).strip()
            except BaseException as exc:  # noqa: BLE001 - a failed call is data, not a crash
                status = STATUS_FAILED
                error = format_error(exc)

            record = build_generation_record(
                scenario=scenario,
                dataset_index=dataset_index,
                generation_index=generation_index,
                seed=seed,
                prompt_text=prompt_text,
                prompt_token_count=prompt_token_count,
                status=status,
                generated_answer=answer,
                generated_token_count=token_count,
                latency=time.time() - gen_start,
                reached_max_tokens=(token_count >= MAX_NEW_TOKENS) if status == STATUS_SUCCESS else False,
                error=error,
            )
            prompt_records.append(record)
            state["generations"].append(record)
            state["pilot_metadata"]["generations_attempted"] = len(state["generations"])
            state["pilot_metadata"]["successful_generations"] = sum(
                1 for g in state["generations"] if g["status"] == STATUS_SUCCESS
            )
            state["pilot_metadata"]["failed_generations"] = (
                state["pilot_metadata"]["generations_attempted"]
                - state["pilot_metadata"]["successful_generations"]
            )

            # FIX 7: persist after every single generation.
            flush()

            log("  gen {0}/{1} seed={2} {3} {4:.2f}s tokens={5}{6}".format(
                generation_index, GENERATIONS_PER_PROMPT, seed, status,
                record["latency"], token_count,
                "" if record["generated_answer"] else " [empty output]" if status == STATUS_SUCCESS else ""))

        # FIX 6: measured diversity for this prompt.
        prompt_metrics = diversity_metrics(prompt_records)
        prompt_metrics["sample_id"] = sample_id
        prompt_metrics["dataset_index"] = dataset_index
        prompt_metrics["answerability"] = answerability
        prompt_metrics["truncation_rate"] = truncation_rate(prompt_records)
        state["diversity_by_prompt"].append(prompt_metrics)
        state["diversity_overall"] = summarize_diversity(state["diversity_by_prompt"])

        # FIX 5 + FIX 8: protected same-seed reproducibility, stored separately.
        try:
            torch.manual_seed(SEEDS[0])
            repro_start = time.time()
            with torch.no_grad():
                repro_outputs = model.generate(
                    **encoded,
                    do_sample=DO_SAMPLE,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    top_k=TOP_K,
                    max_new_tokens=MAX_NEW_TOKENS,
                    pad_token_id=pad_token_id,
                )
            repro_answer = tokenizer.decode(
                repro_outputs[0][prompt_token_count:], skip_special_tokens=True
            ).strip()
            original = next(
                (r["generated_answer"] for r in prompt_records if r.get("seed") == SEEDS[0]), ""
            )
            state["reproducibility"].append(
                {
                    "sample_id": sample_id,
                    "dataset_index": dataset_index,
                    "seed": SEEDS[0],
                    "status": STATUS_SUCCESS,
                    "match": repro_answer == original,
                    "original_output": original,
                    "reproduced_output": repro_answer,
                    "latency": time.time() - repro_start,
                    "error": None,
                    "note": "stored separately; excluded from the primary generation set",
                }
            )
            log("  reproducibility seed={0} match={1}".format(SEEDS[0], repro_answer == original))
        except BaseException as exc:  # noqa: BLE001
            state["reproducibility"].append(
                {
                    "sample_id": sample_id,
                    "dataset_index": dataset_index,
                    "seed": SEEDS[0],
                    "status": STATUS_FAILED,
                    "match": None,
                    "original_output": None,
                    "reproduced_output": None,
                    "latency": None,
                    "error": format_error(exc),
                    "note": "stored separately; excluded from the primary generation set",
                }
            )
            log("  reproducibility FAILED: {0}".format(format_error(exc)))

        state["pilot_metadata"]["prompts_completed"] = len(state["diversity_by_prompt"])
        flush()

    state["pilot_metadata"]["total_generations"] = len(state["generations"])
    state["diversity_overall"] = summarize_diversity(state["diversity_by_prompt"])
    state["pilot_metadata"]["run_state"] = (
        RUN_COMPLETED if state["pilot_metadata"]["prompts_completed"] == len(selected) else RUN_FAILED
    )
    if state["pilot_metadata"]["run_state"] == RUN_FAILED and not state["pilot_metadata"]["run_error"]:
        state["pilot_metadata"]["run_error"] = "not all selected prompts completed"

    persisted = verify_artifact(PILOT_ARTIFACT_PATH, len(state["generations"]))
    flush(artifact_persisted=persisted)

    state["verdict"] = evaluate_verdict(
        run_state=state["pilot_metadata"]["run_state"],
        generations=state["generations"],
        diversity_by_prompt=state["diversity_by_prompt"],
        diversity_overall=state["diversity_overall"],
        reproducibility=state["reproducibility"],
        artifact_persisted=persisted,
    )
    # Verdict is derived, so persist and report from the same object.
    flush(artifact_persisted=persisted)

    PILOT_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    PILOT_REPORT_PATH.write_text(render_report(state), encoding="utf-8")

    verdict = state["verdict"]["verdict"]
    log("CP4.2 pilot finished with verdict {0}".format(verdict))
    log("Report written to {0}".format(PILOT_REPORT_PATH))
    if verdict == VERDICT_PASS:
        return 0
    if verdict == VERDICT_NEEDS_REVISION:
        return 2
    return 1


def main() -> int:
    return run_generation_pilot()


if __name__ == "__main__":
    import sys

    sys.exit(main())
