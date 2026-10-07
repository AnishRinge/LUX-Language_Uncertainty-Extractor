"""Production dataset generation pipeline for LUX.

This pipeline implements the complete end-to-end flow:

    PROMPT  ->  pre-generation Qwen3 hidden representation
           ->  M Qwen3 generations per prompt
           ->  Gemini evaluation (separate phase)
           ->  generation-level labels
           ->  empirical hallucination risk

Parametric: no production dataset size is hardcoded (experiment.yaml has
``generation_count: null``; the plan estimates ~10K-15K as NOT FINAL).
``--num-prompts`` controls selection; ``None`` means "all".

Three phases can run independently so Gemini quota constraints don't block
Qwen3 generation:

    1. generate   - Qwen3 generates answers + captures hidden states
    2. evaluate   - Gemini evaluates each generation (needs API key)
    3. build      - Compute empirical risk + predictor-feature artifact

Frozen constants are inherited from ``run_cp4_2_pilot`` (model revision,
decoding config, thinking=False, CPU_ONLY, seeds).  The canonical scenarios
at ``data/prompts/squad_v2_canonical_scenarios.json`` are read-only input.

Key invariants:
  * The predictor-feature artifact contains ONLY pre-generation information.
    Generated answers, labels, evaluator output, and post-generation
    information are never included.
  * A failed technical operation never becomes a hallucination label.
  * Empirical risk is never fabricated when evaluations are unavailable.
"""

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch  # noqa: E402 - torch is a runtime dependency for generation

# Reuse frozen CP4.2 infrastructure
import run_cp4_2_pilot as cp42

# Reuse evaluation infrastructure
from src.evaluation.evaluator_runner import (
    build_request_from_cp4_2_5,
    compute_empirical_risk,
)
from src.evaluation.gemini_evaluator import GeminiEvaluator
from src.evaluation.schemas import EvaluationRecord, EvaluationStatus, GenerationLabel
from src.utils.hashing import compute_sha256

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ---------------------------------------------------------------------------
# Frozen constants (inherited from CP4.2)
# ---------------------------------------------------------------------------
MODEL_NAME = cp42.MODEL_NAME
MODEL_REVISION = cp42.MODEL_REVISION
DEVICE = cp42.DEVICE
THINKING_MODE = cp42.THINKING_MODE
SEEDS = cp42.SEEDS
DO_SAMPLE = cp42.DO_SAMPLE
TEMPERATURE = cp42.TEMPERATURE
TOP_P = cp42.TOP_P
TOP_K = cp42.TOP_K
MAX_NEW_TOKENS = cp42.MAX_NEW_TOKENS
STATUS_SUCCESS = cp42.STATUS_SUCCESS
STATUS_FAILED = cp42.STATUS_FAILED
CANONICAL_SCENARIOS_PATH = cp42.CANONICAL_SCENARIOS_PATH
SYSTEM_PROMPT = cp42.SYSTEM_PROMPT
DEFAULT_GENERATIONS_PER_PROMPT = len(SEEDS)

# Qwen3 stop-token ids (consistent with CP4.2.1 diagnostic).
EOS_TOKEN_IDS = (151645, 151643)

# ---------------------------------------------------------------------------
# Production artifact paths
# ---------------------------------------------------------------------------
PROD_DIR = Path("data/production")
GENERATIONS_PATH = PROD_DIR / "generations.json"
HIDDEN_STATES_DIR = PROD_DIR / "hidden_states"
PREDICTOR_FEATURES_PATH = PROD_DIR / "predictor_features.json"
EVALUATIONS_DIR = PROD_DIR / "evaluations"
TARGETS_PATH = PROD_DIR / "targets.json"
CHECKPOINT_PATH = PROD_DIR / "checkpoint.json"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Atomic / crash-safe persistence
# ---------------------------------------------------------------------------
def atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    """Write JSON atomically: temp file + fsync + os.replace.

    A reader always observes either the previous complete file or the new
    complete file — never a half-written one.  Same pattern as CP4.2's
    ``write_artifact`` and ``evaluator_runner._atomic_write_json``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False, default=str)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Phase 1: Prompt selection (deterministic, parametric, stratified)
# ---------------------------------------------------------------------------
def load_scenarios(path: Path = CANONICAL_SCENARIOS_PATH) -> List[Dict[str, Any]]:
    """Load canonical SQuAD v2 scenarios from JSON."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def select_production_scenarios(
    scenarios: List[Dict[str, Any]],
    num_prompts: Optional[int] = None,
) -> List[Tuple[int, Dict[str, Any]]]:
    """Select prompts deterministically for production.

    *num_prompts* is parametric — ``None`` selects all scenarios.  When a
    finite count is given, selection is stratified by answerability class
    via CP4.2's ``evenly_spaced_indices`` to preserve the documented
    proportion of answerable vs. unanswerable items.

    Returns ``(dataset_index, scenario)`` tuples where ``dataset_index`` is
    the stable position in the canonical scenarios file.
    """
    total = len(scenarios)
    if num_prompts is None or num_prompts >= total:
        return [(i, s) for i, s in enumerate(scenarios)]

    n_answerable = sum(1 for s in scenarios if cp42.is_answerable(s))
    n_unanswerable = total - n_answerable

    if total == 0:
        return []

    n_ans = round(num_prompts * n_answerable / total)
    n_unans = num_prompts - n_ans
    n_ans = min(n_ans, n_answerable)
    n_unans = min(n_unans, n_unanswerable)

    # Handle tiny requests where proportional split gives 0 of one class.
    if num_prompts > 0 and n_ans == 0 and n_unanswerable > 0:
        n_ans = 1
        n_unans = min(n_unanswerable, num_prompts - 1)
    if num_prompts > 0 and n_unans == 0 and n_answerable > 0:
        n_unans = 1
        n_ans = min(n_answerable, num_prompts - 1)

    return cp42.select_pilot_scenarios(
        scenarios,
        n_answerable=n_ans,
        n_unanswerable=n_unans,
    )


# ---------------------------------------------------------------------------
# Phase 1b: Pre-generation hidden-state capture
# ---------------------------------------------------------------------------
def capture_hidden_states(
    model,
    tokenizer,
    prompt_text: str,
) -> Dict[str, Any]:
    """Forward pass (no generation) to extract pre-generation hidden states.

    Captured BEFORE any ``model.generate()`` call so representations reflect
    the model's state prior to producing an answer.  Stores every layer's
    last-token and mean-pooled representation (layer selection strategy is
    UNRESOLVED per the research plan, so all layers are saved).
    """
    encoded = tokenizer(prompt_text, return_tensors="pt").to(DEVICE)
    seq_len = int(encoded["input_ids"].shape[1])

    with torch.no_grad():
        outputs = model(**encoded, output_hidden_states=True)

    hidden_states = outputs.hidden_states  # tuple: (embedding, layer_0, ..., layer_N)
    num_layers = len(hidden_states)
    hidden_size = int(hidden_states[0].shape[-1])

    last_token: List[List[float]] = []
    mean_pooled: List[List[float]] = []
    for layer_idx in range(num_layers):
        layer_tensor = hidden_states[layer_idx][0]  # (seq, hidden)
        last_tok = layer_tensor[seq_len - 1, :].to(torch.float32).cpu().numpy()
        mean_tok = layer_tensor.mean(dim=0).to(torch.float32).cpu().numpy()
        last_token.append(last_tok.tolist())
        mean_pooled.append(mean_tok.tolist())

    return {
        "seq_len": seq_len,
        "num_layers": num_layers,
        "hidden_size": hidden_size,
        "last_token": last_token,
        "mean_pooled": mean_pooled,
    }


def _hidden_state_path(dataset_index: int) -> Path:
    """Return the canonical path for a prompt's hidden-state file."""
    return HIDDEN_STATES_DIR / "hs_{0}.json".format(dataset_index)


def save_hidden_states(dataset_index: int, payload: Dict[str, Any]) -> Path:
    """Persist hidden states atomically and return the file path."""
    path = _hidden_state_path(dataset_index)
    payload_with_meta = {
        "prompt_dataset_index": dataset_index,
        "model_name": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "hidden_size": payload["hidden_size"],
        "num_layers": payload["num_layers"],
        "seq_len": payload["seq_len"],
        "last_token": payload["last_token"],
        "mean_pooled": payload["mean_pooled"],
    }
    atomic_write_json(path, payload_with_meta)
    return path


# ---------------------------------------------------------------------------
# Phase 1c: Qwen3 generation (frozen protocol)
# ---------------------------------------------------------------------------
def generate_answer(
    model,
    tokenizer,
    encoded_inputs: Dict[str, Any],
    seed: int,
    pad_token_id: Optional[int] = None,
) -> Tuple[str, int, float, Optional[str]]:
    """Generate one answer with the frozen CP4.2 decoding protocol.

    Returns ``(text, token_count, elapsed_sec, error)``.  On a technical
    failure, ``text`` is empty and ``error`` carries the message — the caller
    sets ``status=FAILED``.  A failed technical operation is NEVER silently
    converted into a hallucination label.
    """
    if pad_token_id is None:
        pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    torch.manual_seed(seed)
    t0 = time.time()
    try:
        with torch.no_grad():
            out = model.generate(
                **encoded_inputs,
                do_sample=DO_SAMPLE,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                top_k=TOP_K,
                max_new_tokens=MAX_NEW_TOKENS,
                pad_token_id=pad_token_id,
            )
        elapsed = time.time() - t0
        # transformers >=5 returns a bare Tensor from model.generate(); older
        # versions return a ModelOutput with a .sequences attribute. Accept both
        # without altering the frozen decoding/seed semantics.
        seq = out.sequences[0] if hasattr(out, "sequences") else out[0]
        prompt_len = int(encoded_inputs["input_ids"].shape[1])
        gen_ids = seq[prompt_len:]
        text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        token_count = int(gen_ids.numel())
        return text, token_count, elapsed, None
    except Exception as exc:
        elapsed = time.time() - t0
        return "", 0, elapsed, str(exc)


# ---------------------------------------------------------------------------
# Phase 1d: Build production generation record (extends cp42's frozen record)
# ---------------------------------------------------------------------------
def build_production_record(
    *,
    scenario: Dict[str, Any],
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
    error: Optional[str] = None,
    hidden_state_ref: Optional[str] = None,
    question_token_count: int = 0,
    context_token_count: int = 0,
) -> Dict[str, Any]:
    """Build a production generation record, extending cp42's frozen record.

    The base record (model, revision, decoding config, provenance) is built by
    ``cp42.build_generation_record``.  We add production-specific fields for
    later phases.  No hallucination label or empirical risk is attached here —
    those are added only after evaluation, preventing leakage into features.
    """
    record = cp42.build_generation_record(
        scenario=scenario,
        dataset_index=dataset_index,
        generation_index=generation_index,
        seed=seed,
        prompt_text=prompt_text,
        prompt_token_count=prompt_token_count,
        status=status,
        generated_answer=generated_answer,
        generated_token_count=generated_token_count,
        latency=latency,
        reached_max_tokens=reached_max_tokens,
        error=error,
    )
    source = scenario.get("source", {})
    prompt = scenario.get("prompt", {})
    reference = scenario.get("reference", {})

    record["prompt_id"] = record["sample_id"]
    record["dataset_source"] = source.get("dataset", "squad_v2")
    record["question"] = prompt.get("question", "")
    record["context"] = prompt.get("context", "")
    record["answerability"] = reference.get("answerability", "")
    record["generation_id"] = "{0}:{1}:{2}".format(
        record["sample_id"], dataset_index, seed
    )
    record["generation_status"] = status
    record["hidden_state_reference"] = hidden_state_ref
    record["question_token_count"] = question_token_count
    record["context_token_count"] = context_token_count
    record["evaluation_status"] = None
    record["generation_label"] = None
    record["claim_evaluations"] = None
    record["empirical_hallucination_risk"] = None
    return record


# ---------------------------------------------------------------------------
# Phase 1: Generation phase (resumable)
# ---------------------------------------------------------------------------
def load_checkpoint() -> Dict[str, Any]:
    """Load or initialise the run checkpoint."""
    if CHECKPOINT_PATH.exists():
        try:
            with open(CHECKPOINT_PATH, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except (json.JSONDecodeError, OSError):
            pass
    return {
        "run_id": hashlib.sha256(str(time.time()).encode("utf-8")).hexdigest()[:16],
        "phase": "not_started",
        "started_at": _now_iso(),
        "last_updated": _now_iso(),
        "num_prompts": None,
        "generations_per_prompt": DEFAULT_GENERATIONS_PER_PROMPT,
        "model_revision": MODEL_REVISION,
        "completed_prompts": [],
        "all_generations": [],
        "selected_scenarios": [],
    }


def save_checkpoint(checkpoint: Dict[str, Any]) -> None:
    checkpoint["last_updated"] = _now_iso()
    atomic_write_json(CHECKPOINT_PATH, checkpoint)


def run_generation_phase(
    num_prompts: Optional[int] = None,
    generations_per_prompt: Optional[int] = None,
    output_dir: Path = PROD_DIR,
) -> Dict[str, Any]:
    """Phase 1: capture hidden states + Qwen3 generations.

    Resumable: prompts already in ``checkpoint["completed_prompts"]`` are
    skipped.  The checkpoint is written atomically after each prompt so a
    crash never loses completed work.
    """
    gens_per_prompt = (
        generations_per_prompt
        if generations_per_prompt
        else DEFAULT_GENERATIONS_PER_PROMPT
    )
    seeds = list(SEEDS[:gens_per_prompt])

    checkpoint = load_checkpoint()
    checkpoint["phase"] = "generation"
    checkpoint["num_prompts"] = num_prompts
    checkpoint["generations_per_prompt"] = gens_per_prompt
    save_checkpoint(checkpoint)

    scenarios = load_scenarios()
    selected = select_production_scenarios(scenarios, num_prompts)
    checkpoint["selected_scenarios"] = [
        {"dataset_index": idx, "sample_id": s.get("sample_id")}
        for idx, s in selected
    ]

    completed: set = set(checkpoint.get("completed_prompts", []))
    all_records: List[Dict[str, Any]] = list(
        checkpoint.get("all_generations", [])
    )

    print(
        "Phase 1 (generation): {0} prompts, {1} generations/prompt, device={2}".format(
            len(selected), gens_per_prompt, DEVICE
        ),
        flush=True,
    )

    tokenizer, model = cp42._load_model()
    model.eval()
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    prompt_t0 = time.time()
    for dataset_index, scenario in selected:
        if dataset_index in completed:
            print("  skip prompt {0} (already complete)".format(dataset_index), flush=True)
            continue

        prompt_text = cp42.render_prompt(
            tokenizer,
            scenario["prompt"]["context"],
            scenario["prompt"]["question"],
        )
        encoded = tokenizer(
            prompt_text, return_tensors="pt"
        ).to(DEVICE)
        prompt_token_count = int(encoded["input_ids"].shape[1])

        # --- Pre-generation hidden states (BEFORE any generate() call) ---
        hidden_state_ref: Optional[str] = None
        try:
            hidden_payload = capture_hidden_states(model, tokenizer, prompt_text)
            hidden_path = save_hidden_states(dataset_index, hidden_payload)
            hidden_state_ref = str(hidden_path)
        except Exception as exc:
            print(
                "  WARNING: hidden-state capture failed for prompt {0}: {1}".format(
                    dataset_index, exc
                ),
                flush=True,
            )

        # --- Generate M answers (frozen protocol) ---
        q_tokens = len(tokenizer(scenario["prompt"]["question"], add_special_tokens=False)["input_ids"])
        c_tokens = len(tokenizer(scenario["prompt"]["context"], add_special_tokens=False)["input_ids"])

        for gen_index, seed in enumerate(seeds):
            text, token_count, elapsed, error = generate_answer(
                model, tokenizer, encoded, seed, pad_token_id
            )
            if text and not error:
                status = STATUS_SUCCESS
            else:
                status = STATUS_FAILED

            record = build_production_record(
                scenario=scenario,
                dataset_index=dataset_index,
                generation_index=gen_index,
                seed=seed,
                prompt_text=prompt_text,
                prompt_token_count=prompt_token_count,
                status=status,
                generated_answer=text,
                generated_token_count=token_count,
                latency=elapsed,
                reached_max_tokens=token_count >= MAX_NEW_TOKENS,
                error=error,
                hidden_state_ref=hidden_state_ref,
                question_token_count=q_tokens,
                context_token_count=c_tokens,
            )
            all_records.append(record)

        completed.add(dataset_index)
        checkpoint["completed_prompts"] = sorted(completed)
        checkpoint["all_generations"] = all_records
        save_checkpoint(checkpoint)

        prompt_elapsed = time.time() - prompt_t0
        print(
            "  prompt {0}: {1} generations, {2}s (cumulative: {3})".format(
                dataset_index, gens_per_prompt, prompt_elapsed, len(all_records)
            ),
            flush=True,
        )
        prompt_t0 = time.time()

    artifact = {
        "pipeline": {
            "checkpoint": "CP5 production dataset",
            "run_id": checkpoint["run_id"],
            "started_at": checkpoint["started_at"],
            "completed_at": _now_iso(),
            "phase": "generation_complete",
            "num_prompts": len(selected),
            "generations_per_prompt": gens_per_prompt,
            "seeds": list(seeds),
        },
        "model": {
            "name": MODEL_NAME,
            "revision": MODEL_REVISION,
            "device": DEVICE,
            "thinking_mode": THINKING_MODE,
            "decoding": {
                "do_sample": DO_SAMPLE,
                "temperature": TEMPERATURE,
                "top_p": TOP_P,
                "top_k": TOP_K,
                "max_new_tokens": MAX_NEW_TOKENS,
            },
        },
        "environment": {
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "device": DEVICE,
        },
        "scenarios": checkpoint["selected_scenarios"],
        "generations": all_records,
        "summary": {
            "n_prompts": len(selected),
            "n_generations": len(all_records),
            "n_success": sum(1 for r in all_records if r["generation_status"] == STATUS_SUCCESS),
            "n_failed": sum(1 for r in all_records if r["generation_status"] == STATUS_FAILED),
            "n_with_hidden_states": sum(1 for r in all_records if r["hidden_state_reference"]),
        },
    }
    atomic_write_json(GENERATIONS_PATH, artifact)
    print("Generations artifact written to {0}".format(GENERATIONS_PATH), flush=True)

    checkpoint["phase"] = "generation_complete"
    save_checkpoint(checkpoint)
    return artifact


# ---------------------------------------------------------------------------
# Phase 2: Evaluation phase (resumable, uses Gemini default retry policy)
# ---------------------------------------------------------------------------
def _generation_evaluation_path(generation_id: str) -> Path:
    safe = generation_id.replace(":", "_").replace("/", "_")
    return EVALUATIONS_DIR / "{0}.json".format(safe)


def _eval_requests_from_generations(
    generations: List[Dict[str, Any]],
    scenarios: Dict[int, Dict[str, Any]],
) -> List[Tuple[Dict[str, Any], Any]]:
    """Build evaluation requests for generations not yet evaluated."""
    pending: List[Tuple[Dict[str, Any], Any]] = []
    for gen in generations:
        if gen.get("evaluation_status") is not None:
            continue
        sid = gen["dataset_index"]
        scenario = scenarios.get(sid)
        if scenario is None:
            continue
        req = build_request_from_cp4_2_5(gen, scenario)
        pending.append((gen, req))
    return pending


def run_evaluation_phase(
    api_key: Optional[str] = None,
    spacing_seconds: float = 12.0,
) -> None:
    """Phase 2: evaluate each generation with Gemini.

    Uses the **normal** GeminiEvaluator retry policy (max_retries=3).  Calls
    are spaced by *spacing_seconds* to respect free-tier rate limits.  Already-
    evaluated generations are skipped (resume).
    """
    if not GENERATIONS_PATH.exists():
        raise SystemExit(
            "Generation artifact not found: {0}. Run 'generate' first.".format(
                GENERATIONS_PATH
            )
        )

    with open(GENERATIONS_PATH, "r", encoding="utf-8") as fh:
        artifact = json.load(fh)

    scenarios: Dict[int, Dict[str, Any]] = {}
    for s in artifact.get("scenarios", []):
        scenarios[s["dataset_index"]] = s
    # Merge full scenario data (question, context) for request building.
    full_scenarios = dict(enumerate(load_scenarios()))
    for ds_idx in list(scenarios.keys()):
        if ds_idx in full_scenarios:
            scenarios[ds_idx] = full_scenarios[ds_idx]

    pending = _eval_requests_from_generations(artifact["generations"], scenarios)
    if not pending:
        print("Phase 2 (evaluation): all generations already evaluated.", flush=True)
        return

    print(
        "Phase 2 (evaluation): {0} pending generations (of {1} total).".format(
            len(pending), len(artifact["generations"])
        ),
        flush=True,
    )

    evaluator = GeminiEvaluator(
        api_key=api_key,
        retry_policy=None,  # use GeminiEvaluator's default RetryPolicy(max_retries=3)
    )

    for idx, (gen, req) in enumerate(pending):
        record = evaluator.evaluate(req)
        record_dict = record.to_dict()

        eval_path = _generation_evaluation_path(gen["generation_id"])
        atomic_write_json(eval_path, record_dict)

        gen["evaluation_status"] = record.evaluation_status.value
        gen["generation_label"] = (
            record.generation_label.value
            if record.generation_label is not None
            else None
        )
        gen["claim_evaluations"] = record_dict.get("claims")

        if idx < len(pending) - 1 and spacing_seconds > 0:
            time.sleep(spacing_seconds)

        print(
            "  [{0}/{1}] {2}: status={3} label={4}".format(
                idx + 1,
                len(pending),
                gen["generation_id"],
                gen["evaluation_status"],
                gen["generation_label"] or "-",
            ),
            flush=True,
        )

    atomic_write_json(GENERATIONS_PATH, artifact)
    print("Updated generations artifact with evaluations.", flush=True)


# ---------------------------------------------------------------------------
# Phase 3: Target computation + predictor-feature separation
# ---------------------------------------------------------------------------
def _group_generations_by_prompt(
    generations: List[Dict[str, Any]],
) -> Dict[int, List[Dict[str, Any]]]:
    """Group generation records by ``dataset_index``."""
    groups: Dict[int, List[Dict[str, Any]]] = {}
    for gen in generations:
        key = gen["dataset_index"]
        groups.setdefault(key, []).append(gen)
    return groups


def compute_per_prompt_risk(
    eval_records: List[EvaluationRecord],
) -> Dict[str, Any]:
    """Compute empirical hallucination risk for a set of evaluation records.

    Reuses the frozen formula from ``evaluator_runner.compute_empirical_risk``:

        empirical_hallucination_risk =
            UNRELIABLE generations (evaluation_status == SUCCESS)
          / (generations with evaluation_status == SUCCESS)

    FAILED and MANUAL_REVIEW evaluations are excluded from the denominator.
    If no evaluations have SUCCESS status, risk is ``None`` (never fabricated).
    """
    result = compute_empirical_risk(eval_records)
    return {
        "empirical_hallucination_risk": result["empirical_hallucination_risk"],
        "n_resolved": result["total_resolved"],
        "n_unreliable": result["unreliable_resolved"],
        "status_counts": result["status_counts"],
        "label_counts": result["label_counts"],
    }


def build_targets(
    generations: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Compute per-prompt empirical risk from evaluation records.

    Groups generations by ``dataset_index`` and, for each prompt, loads
    evaluation records from ``evaluations/`` and computes the frozen risk ratio.

    A prompt with zero SUCCESS evaluations gets ``risk = None`` (never
    fabricated).  Failed technical operations are never labelled as
    hallucination: FAILED evaluations carry no ``generation_label`` and are
    excluded from the denominator.
    """
    groups = _group_generations_by_prompt(generations)
    targets: List[Dict[str, Any]] = []

    for dataset_index, prompt_gens in groups.items():
        eval_records: List[EvaluationRecord] = []
        for gen in prompt_gens:
            eval_path = _generation_evaluation_path(gen["generation_id"])
            if eval_path.exists():
                with open(eval_path, "r", encoding="utf-8") as fh:
                    eval_dict = json.load(fh)
                try:
                    eval_records.append(EvaluationRecord.from_dict(eval_dict))
                except ValueError:
                    pass  # skip malformed records

        risk_info = compute_per_prompt_risk(eval_records)
        risk_value = risk_info["empirical_hallucination_risk"]

        for gen in prompt_gens:
            gen["empirical_hallucination_risk"] = risk_value

        first = prompt_gens[0]
        targets.append({
            "prompt_id": first["prompt_id"],
            "dataset_index": dataset_index,
            "dataset_source": first["dataset_source"],
            "answerability": first["answerability"],
            "n_generations": len(prompt_gens),
            **{k: v for k, v in risk_info.items()},
        })

    return targets


def build_predictor_features(
    generations: List[Dict[str, Any]],
    tokenizer=None,
) -> Dict[str, Any]:
    """Build a predictor-feature artifact containing ONLY pre-generation info.

    This artifact is the **exclusive** input for the predictor.  It is
    deliberately stripped of every post-generation field:

    Excluded: ``generated_answer``, ``generation_label``, ``claim_evaluations``,
    ``evaluation_status``, ``empirical_hallucination_risk``, ``status``,
    ``latency``, ``generated_token_count``, ``dataset_source``.

    ``dataset_source`` is excluded per the leakage constraints
    (source_dataset must not be a predictor feature).  Each entry links to
    its hidden-state file via ``hidden_state_ref``.
    """
    seen: Dict[int, Dict[str, Any]] = {}
    for gen in generations:
        if gen["dataset_index"] in seen:
            continue

        q = gen["question"]
        ctx = gen["context"]
        metrics: Dict[str, Any] = {
            "question_chars": len(q),
            "context_chars": len(ctx),
            "question_ends_with_question_mark": q.rstrip().endswith("?"),
            "answerability": gen["answerability"],
        }
        if tokenizer is not None:
            metrics["question_tokens"] = len(
                tokenizer(q, add_special_tokens=False)["input_ids"]
            )
            metrics["context_tokens"] = len(
                tokenizer(ctx, add_special_tokens=False)["input_ids"]
            )

        seen[gen["dataset_index"]] = {
            "prompt_id": gen["prompt_id"],
            "dataset_index": gen["dataset_index"],
            "answerability": gen["answerability"],
            "hidden_state_ref": gen["hidden_state_reference"],
            "prompt_metrics": metrics,
        }

    excluded = [
        "generated_answer",
        "generation_label",
        "claim_evaluations",
        "evaluation_status",
        "empirical_hallucination_risk",
        "status",
        "latency",
        "generated_token_count",
        "reached_max_tokens",
        "dataset_source",
    ]
    features = {
        "pipeline": {
            "artifact": "predictor_features",
            "created_at": _now_iso(),
            "model_revision": MODEL_REVISION,
            "excluded_fields": excluded,
        },
        "predictor_features": list(seen.values()),
    }
    atomic_write_json(PREDICTOR_FEATURES_PATH, features)
    return features


def load_tokenizer():
    """Load only the tokenizer (avoids loading the full model)."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        MODEL_NAME, revision=MODEL_REVISION, trust_remote_code=True
    )


def build_targets_and_features(
    force_rebuild_predictor: bool = False,
) -> Dict[str, Any]:
    """Phase 3: compute per-prompt risk + predictor-feature artifact.

    This phase only reads already-evaluated generations.  If no evaluations
    are available, risk values are ``None`` (never fabricated).
    """
    if not GENERATIONS_PATH.exists():
        raise SystemExit(
            "Generations artifact not found: {0}. Run 'generate' first.".format(
                GENERATIONS_PATH
            )
        )

    with open(GENERATIONS_PATH, "r", encoding="utf-8") as fh:
        artifact = json.load(fh)

    generations = artifact["generations"]

    # Compute targets (risk) from evaluation records on disk.
    targets = build_targets(generations)
    atomic_write_json(TARGETS_PATH, {"targets": targets})
    print("Targets written to {0}".format(TARGETS_PATH), flush=True)

    # Write updated artifact with risk values.
    atomic_write_json(GENERATIONS_PATH, artifact)

    # Build predictor features (only if not already built or forced).
    if PREDICTOR_FEATURES_PATH.exists() and not force_rebuild_predictor:
        print("Predictor features already exist; skipping (use --force-rebuild to refresh).", flush=True)
    else:
        tokenizer = load_tokenizer()
        features = build_predictor_features(generations, tokenizer)
        print(
            "Predictor features written to {0} ({1} prompts)".format(
                PREDICTOR_FEATURES_PATH, len(features["predictor_features"])
            ),
            flush=True,
        )

    return {
        "n_targets": len(targets),
        "n_predictor_features": len(features["predictor_features"]) if not (
            PREDICTOR_FEATURES_PATH.exists() and not force_rebuild_predictor
        ) else 0,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="LUX production dataset generation pipeline"
    )
    subparsers = parser.add_subparsers(dest="phase")

    gen_p = subparsers.add_parser(
        "generate", help="Phase 1: Qwen3 generations + hidden states"
    )
    gen_p.add_argument(
        "--num-prompts", type=int, default=None,
        help="Number of prompts to select (default: all)",
    )
    gen_p.add_argument(
        "--generations-per-prompt", type=int, default=None,
        help="Generations per prompt (default: {0})".format(
            DEFAULT_GENERATIONS_PER_PROMPT
        ),
    )

    eval_p = subparsers.add_parser(
        "evaluate", help="Phase 2: Gemini evaluation"
    )
    eval_p.add_argument(
        "--spacing", type=float, default=12.0,
        help="Seconds between Gemini calls (default: 12.0)",
    )

    build_p = subparsers.add_parser(
        "build", help="Phase 3: targets + predictor features"
    )
    build_p.add_argument(
        "--force-rebuild-predictor", action="store_true",
        help="Rebuild predictor features even if they exist",
    )

    args = parser.parse_args(argv)

    if args.phase == "generate":
        run_generation_phase(args.num_prompts, args.generations_per_prompt)
    elif args.phase == "evaluate":
        run_evaluation_phase(spacing_seconds=args.spacing)
    elif args.phase == "build":
        build_targets_and_features(
            force_rebuild_predictor=args.force_rebuild_predictor
        )
    else:
        parser.print_help()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
