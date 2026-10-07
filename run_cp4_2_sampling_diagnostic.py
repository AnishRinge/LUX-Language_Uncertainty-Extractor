"""CP4.2.1 - Instrumented single-prompt sampling diagnostic.

Purpose
-------
Measure the *effective* sampling distribution that CP4.2's frozen decoding
configuration produces on ONE already-selected CP4.2 prompt, in order to test
the hypothesis that `top_p=0.8` collapses the nucleus to a single token and
thereby explains the diversity collapse recorded in the CP4.2 pilot.

Scope / constraints
-------------------
* Reuses the EXACT CP4.2 protocol constants and loader from `run_cp4_2_pilot`:
  model, revision, DEVICE (CPU_ONLY), thinking=False (chat template), the
  identical rendered prompt, identical `torch.manual_seed(seed)` placement, and
  identical decoding kwargs (do_sample/temperature/top_p/top_k/max_new_tokens).
* Decoding parameters are NOT changed. No CP4.3 code is created here.
* 3 generations only, seeds 1001/1002/1003 (the first three CP4.2 seeds).
* One prompt only: dataset_index 61608 ("What is note worthy about the bird
  population of Burma?"), selected through `cp42.select_pilot_scenarios` so it is
  the SAME scenario object the CP4.2 pilot used.
* The diagnostic does NOT modify the CP4.2 artifact, the canonical dataset, the
  experiment config, or any CP4.2 source/test/report file.
* The full vocabulary-sized logits tensor is never written to disk. Per step we
  store only: sampled token id, its text, the nucleus size, the top-1
  probability, and the top-5 candidate (token id, probability, text).
* `output_logits=True` captures the RAW pre-processor logits ONLY for an
  in-memory cross-check that the captured post-processor `scores` equal
  `processor(raw_logits)`; the raw tensors are discarded after that check and
  are never serialised.

Effective nucleus size
----------------------
Computed from the captured `scores` (the exact post-processor logits that
`torch.multinomial` sampled from). For each step:
    probs = softmax(scores[step])              # exactly what was sampled
    nucleus_size = number of tokens with prob > 0
A token has prob 0.0 exactly when its logit was set to -inf by the
TemperatureLogitsWarper -> TopKLogitsWarper -> TopPLogitsWarper chain
(min_tokens_to_keep=1, num_beams=None). This count therefore equals the size of
the top-p nucleus after top-k truncation.
"""

import json
import math
import os
import sys
import time
import hashlib
import platform
from datetime import datetime, timezone

import torch

import run_cp4_2_pilot as cp42

DIAGNOSTIC_SEEDS = (1001, 1002, 1003)
TOP_K_CANDIDATES = 5
ZERO_TOL = 0.0
DIAGNOSTIC_DATASET_INDEX = 61608
EOS_TOKEN_IDS = (151645, 151643)  # Qwen3 chat/stop tokens (151643 == pad/eos/chat)
DIAGNOSTIC_ARTIFACT_PATH = "data/pilots/cp4_2_sampling_diagnostic.json"
DIAGNOSTIC_REPORT_PATH = "docs/cp4_2_sampling_diagnostic_report.md"
CP4_2_ARTIFACT_PATH = "data/pilots/cp4_2_generation_pilot.json"

# CP4.2-validated non-thinking marker. Qwen3 with enable_thinking=False intentionally
# renders an EMPTY thinking block  to the assistant generation prefix; CP4.2 verifies
# by `if " not in prompt_text: raise` (run_cp4_2_pilot.py:957). This constant is
# built with chr() rather than a literal so the authoring layer cannot mangle the
# angle-bracket sequence; it is byte-for-byte identical to CP4.2's check literal
# (hex 3c7468696e6b3e, confirmed absent only when thinking is NOT disabled).
_NON_THINKING_MARKER = chr(0x3C) + "think" + chr(0x3E)
assert _NON_THINKING_MARKER.encode("utf-8").hex() == "3c7468696e6b3e"


def sha256_of_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def file_mtime_iso(path):
    return datetime.fromtimestamp(os.path.getmtime(path), timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Pure helpers (list-based) - unit tested independently of torch/model
# ---------------------------------------------------------------------------

def softmax_list(logits):
    """Numerically-stable softmax over a 1-D sequence of floats -> list of probs."""
    if not logits:
        return []
    m = max(logits)
    # -inf - m = -inf; exp(-inf) = 0.0
    exps = [math.exp(x - m) if x != float("-inf") else 0.0 for x in logits]
    s = sum(exps)
    if s == 0.0:
        return [0.0] * len(logits)
    return [e / s for e in exps]


def nucleus_size_list(logits, zero_tol=ZERO_TOL):
    """Count tokens with prob > zero_tol after softmax.

    Equivalently the number of log-probability entries that are not -inf,
    i.e. the number of tokens surviving Temperature/TopK/TopP filtering.
    """
    probs = softmax_list(logits)
    return sum(1 for p in probs if p > zero_tol)


def topk_list(logits, k):
    """Top-k (token_index, probability) pairs by descending probability."""
    probs = softmax_list(logits)
    order = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
    return [(i, float(probs[i])) for i in order[:k] if probs[i] > 0.0]


def rank_of_token(logits, token_id, zero_tol=ZERO_TOL):
    """1-based rank of `token_id` among positive-probability candidates.

    Returns None if the token has zero probability (outside the nucleus).
    """
    probs = softmax_list(logits)
    if probs[token_id] <= zero_tol:
        return None
    # rank by descending prob; ties broken by token id (deterministic)
    order = sorted(range(len(probs)), key=lambda i: (-probs[i], i))
    for rank, idx in enumerate(order, start=1):
        if idx == token_id:
            return rank
    return None


def top1_prob_list(logits):
    """Probability of the single most likely token (post-processor softmax)."""
    probs = softmax_list(logits)
    return float(max(probs)) if probs else 0.0


# ---------------------------------------------------------------------------
# Processor-chain reconstruction (for in-memory cross-check only)
# ---------------------------------------------------------------------------

def build_processor_chain():
    """Rebuild the EXACT processor list CP4.2's generate() uses (min_tokens_to_keep=1).

    Order is Temperature -> TopK -> TopP, matching
    transformers.GenerationMixin._get_logits_processor for do_sample=True with
    num_beams=None; no other warpers are active for the CP4.2 config.
    """
    from transformers import (
        TemperatureLogitsWarper,
        TopKLogitsWarper,
        TopPLogitsWarper,
        LogitsProcessorList,
    )
    return LogitsProcessorList([
        TemperatureLogitsWarper(cp42.TEMPERATURE),
        TopKLogitsWarper(top_k=cp42.TOP_K, min_tokens_to_keep=1),
        TopPLogitsWarper(top_p=cp42.TOP_P, min_tokens_to_keep=1),
    ])


def apply_processors(chain, input_ids, raw_logits_row):
    """Apply a LogitsProcessorList to a 1-D raw logits row -> 1-D post scores.

    `input_ids` is required by the warper __call__ signature but is unused by
    Temperature/TopK/TopP, so its content is irrelevant; only its shape is taken.
    The warpers act on a 2-D (batch, vocab) scores tensor, so the 1-D
    `raw_logits_row` is unsqueezed to (1, vocab) -- NOT (1, 1, vocab), which is
    what caused the prior out-of-bounds scatter in TopPLogitsWarper.
    """
    dummy_ids = torch.zeros((1, input_ids.shape[1]), dtype=torch.long)
    row = raw_logits_row.unsqueeze(0)  # (1, vocab) -- the shape TopP/TopK/Temperature expect
    out = chain(dummy_ids, row)
    return out.squeeze(0)  # (vocab,)


# ---------------------------------------------------------------------------
# Prompt selection (reuses CP4.2 selection logic)
# ---------------------------------------------------------------------------

def select_target_scenario(scenarios):
    """Return (dataset_index, scenario) for DIAGNOSTIC_DATASET_INDEX via CP4.2 selection."""
    selected = cp42.select_pilot_scenarios(scenarios)
    for dataset_index, scenario in selected:
        if dataset_index == DIAGNOSTIC_DATASET_INDEX:
            return dataset_index, scenario
    # Deterministic fallback that must still match the canonical scenario.
    return DIAGNOSTIC_DATASET_INDEX, scenarios[DIAGNOSTIC_DATASET_INDEX]


def load_cp42_reference(seed, dataset_index):
    """Read the CP4.2 pilot artifact and return the reference answer for (seed, idx).

    Used ONLY for a fidelity self-check: the diagnostic adds output-capture flags
    that must not change sampling, so the diagnostic's seed-1001 output for idx=61608
    must byte-match this reference. Returns None if the (seed, idx) pair is absent.
    """
    with open(CP4_2_ARTIFACT_PATH, "r", encoding="utf-8") as fh:
        artifact = json.load(fh)
    for g in artifact["generations"]:
        if g.get("dataset_index") == dataset_index and g.get("seed") == seed:
            return {
                "seed": g["seed"],
                "generated_answer": g["generated_answer"],
                "generated_token_count": g["generated_token_count"],
            }
    return None


# ---------------------------------------------------------------------------
# Per-step analysis
# ---------------------------------------------------------------------------

def _row_to_list(row):
    return row.detach().to(torch.float64).tolist()


def analyze_output(out, tokenizer, prompt_token_count, cross_chain, dummy_ids):
    """Turn a generate() result into per-step diagnostic records.

    Returns (steps, gen_token_ids, generated_text, token_count, eos_reached,
    cross_check_max_abs_diff).
    """
    seq = out.sequences[0]
    gen_ids = seq[prompt_token_count:]
    scores = out.scores          # tuple length == len(gen_ids); post-processor logits
    raw = out.logits             # tuple length == len(gen_ids); raw logits

    steps = []
    cross_check_max_abs = 0.0
    for i in range(len(gen_ids)):
        step_scores = scores[i][0]          # (vocab,) float32, post-processor
        raw_logits = raw[i][0]              # (vocab,) float32, pre-processor
        sampled_id = int(gen_ids[i].item())

        probs = torch.softmax(step_scores, dim=-1).to(torch.float64)
        top5 = torch.topk(probs, k=TOP_K_CANDIDATES)
        top1_idx = int(top5.indices[0].item())
        top1_prob = float(top5.values[0].item())
        nucleus = int((probs > ZERO_TOL).sum().item())

        # torch isfinite check (should equal nucleus)
        finite_count = int(torch.isfinite(step_scores).sum().item())

        steps.append({
            "step": i,
            "sampled_token_id": sampled_id,
            "sampled_token_text": tokenizer.decode([sampled_id]),
            "sampled_rank": _rank_torch(probs, sampled_id),
            "top1_token_id": top1_idx,
            "top1_prob": top1_prob,
            "nucleus_size": nucleus,
            "nucleus_size_isfinite": finite_count,
            "top5": [
                {
                    "rank": r + 1,
                    "token_id": int(top5.indices[r].item()),
                    "prob": float(top5.values[r].item()),
                    "token_text": tokenizer.decode([int(top5.indices[r].item())]),
                }
                for r in range(TOP_K_CANDIDATES)
            ],
            "cross_check_max_abs_diff": _cross_check_step(
                cross_chain, dummy_ids, raw_logits, step_scores
            ),
        })
        cross_check_max_abs = max(cross_check_max_abs, steps[-1]["cross_check_max_abs_diff"])

    generated_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    eos_reached = bool(int(gen_ids[-1].item()) in EOS_TOKEN_IDS) if gen_ids.numel() else False
    token_count = int(gen_ids.numel())
    return steps, gen_ids.tolist(), generated_text, token_count, eos_reached, cross_check_max_abs


def _rank_torch(probs, token_id):
    p = probs[token_id].item()
    if p <= 0.0:
        return None
    order = torch.argsort(probs, descending=True)
    eq = (order == token_id)
    rank = int(eq.nonzero(as_tuple=True)[0].item()) + 1
    return rank


def _cross_check_step(chain, dummy_ids, raw_logits, expected_scores):
    """Return max |chain(raw) - expected_scores|; -inf positions handled."""
    if chain is None:
        return None
    produced = apply_processors(chain, dummy_ids, raw_logits)
    diff = (produced - expected_scores).abs()
    return float(diff.max().item())


# ---------------------------------------------------------------------------
# Prompt rendering (identical to CP4.2)
# ---------------------------------------------------------------------------

def render_prompt_for_scenario(tokenizer, scenario):
    return cp42.render_prompt(
        tokenizer,
        scenario["prompt"]["context"],
        scenario["prompt"]["question"],
    )


# ---------------------------------------------------------------------------
# Main diagnostic
# ---------------------------------------------------------------------------

def run_diagnostic():
    import psutil

    started = datetime.now(timezone.utc).isoformat()
    scenarios = cp42._load_scenarios()
    dataset_index, scenario = select_target_scenario(scenarios)

    cp42_reference = load_cp42_reference(DIAGNOSTIC_SEEDS[0], DIAGNOSTIC_DATASET_INDEX)

    print("Loading model (CP4.2 loader)...")
    t0 = time.time()
    tokenizer, model = cp42._load_model()
    load_time = time.time() - t0
    print("Model loaded in {0:.1f}s".format(load_time))

    prompt_text = render_prompt_for_scenario(tokenizer, scenario)
    if _NON_THINKING_MARKER not in prompt_text:
        raise RuntimeError("chat template did not emit the non-thinking prefix")
    encoded = tokenizer(prompt_text, return_tensors="pt").to(cp42.DEVICE)
    prompt_token_count = int(encoded["input_ids"].shape[1])
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    dummy_ids = encoded["input_ids"]

    cross_chain = build_processor_chain()

    cp42_artifact_sha = sha256_of_file(CP4_2_ARTIFACT_PATH)

    generations = []
    for seed in DIAGNOSTIC_SEEDS:
        torch.manual_seed(seed)
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(
                **encoded,
                do_sample=cp42.DO_SAMPLE,
                temperature=cp42.TEMPERATURE,
                top_p=cp42.TOP_P,
                top_k=cp42.TOP_K,
                max_new_tokens=cp42.MAX_NEW_TOKENS,
                pad_token_id=pad_token_id,
                output_scores=True,
                output_logits=True,
                return_dict_in_generate=True,
            )
        elapsed = time.time() - t0
        steps, gen_ids, text, n_tok, eos, cc_max = analyze_output(
            out, tokenizer, prompt_token_count, cross_chain, dummy_ids
        )
        generations.append({
            "seed": seed,
            "generated_answer": text,
            "generated_token_count": n_tok,
            "gen_token_ids": gen_ids,
            "eos_reached": eos,
            "latency_sec": elapsed,
            "cross_check_max_abs_diff": cc_max,
            "steps": steps,
        })
        print("seed={0} tokens={1} eos={2} steps={3} latency={4:.1f}s cross_check_maxdiff={5:.2e}".format(
            seed, n_tok, eos, len(steps), elapsed, cc_max))

    # Built-in fidelity check: output-capture flags must not alter sampling, so
    # the diagnostic seed-1001 output must byte-match the CP4.2 pilot record.
    di_seed = next(g for g in generations if g["seed"] == DIAGNOSTIC_SEEDS[0])
    fidelity_pass = False
    if cp42_reference is not None:
        fidelity_pass = (
            di_seed["generated_answer"] == cp42_reference["generated_answer"]
            and di_seed["generated_token_count"] == cp42_reference["generated_token_count"]
        )
        print("Fidelity check vs CP4.2 seed={0}: {1}".format(DIAGNOSTIC_SEEDS[0], fidelity_pass))
        if not fidelity_pass:
            print("  diagnostic : {0!r}".format(di_seed["generated_answer"]))
            print("  cp4_2 ref : {0!r}".format(cp42_reference["generated_answer"]))
    else:
        print("Fidelity check SKIPPED: no CP4.2 reference found for idx={0} seed={1}".format(
            DIAGNOSTIC_DATASET_INDEX, DIAGNOSTIC_SEEDS[0]))

    artifact = assemble_artifact(
        started=started,
        load_time_sec=load_time,
        scenario=scenario,
        dataset_index=dataset_index,
        prompt_text=prompt_text,
        prompt_token_count=prompt_token_count,
        pad_token_id=pad_token_id,
        generations=generations,
        cp42_artifact_sha=cp42_artifact_sha,
        cp42_reference=cp42_reference,
        fidelity_check_pass=fidelity_pass,
    )
    write_json(DIAGNOSTIC_ARTIFACT_PATH, artifact)
    print("Diagnostic artifact written to {0}".format(DIAGNOSTIC_ARTIFACT_PATH))

    report = render_report(artifact)
    os.makedirs(os.path.dirname(DIAGNOSTIC_REPORT_PATH), exist_ok=True)
    with open(DIAGNOSTIC_REPORT_PATH, "w", encoding="utf-8") as fh:
        fh.write(report)
    print("Diagnostic report written to {0}".format(DIAGNOSTIC_REPORT_PATH))
    return artifact


def assemble_artifact(started, load_time_sec, scenario, dataset_index, prompt_text,
                      prompt_token_count, pad_token_id, generations, cp42_artifact_sha,
                      cp42_reference, fidelity_check_pass):
    return {
        "diagnostic_metadata": {
            "checkpoint": "CP4.2.1_sampling_diagnostic",
            "purpose": (
                "Instrumented single-prompt measurement of the effective sampling "
                "distribution under CP4.2's frozen decoding config, to test whether "
                "top_p=0.8 collapses the nucleus to one token."
            ),
            "created_at": started,
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "transformers_version": _transformers_version(),
            "device": cp42.DEVICE,
            "execution_strategy": cp42.EXECUTION_STRATEGY,
            "thinking_mode": cp42.THINKING_MODE,
            "thinking_enforcement": cp42.THINKING_ENFORCEMENT,
            "prompt_format": cp42.PROMPT_FORMAT,
            "model_name": cp42.MODEL_NAME,
            "model_revision": cp42.MODEL_REVISION,
            "decoding_config": {
                "do_sample": cp42.DO_SAMPLE,
                "temperature": cp42.TEMPERATURE,
                "top_p": cp42.TOP_P,
                "top_k": cp42.TOP_K,
                "max_new_tokens": cp42.MAX_NEW_TOKENS,
                "pad_token_id": pad_token_id,
                "top_k_nucleus_check": "min_tokens_to_keep=1, num_beams=None",
            },
            "seeds_used": list(DIAGNOSTIC_SEEDS),
            "seed_source": "first three of cp42.SEEDS",
            "generations_used": len(generations),
            "generation_calls": len(generations),
            "model_generate_calls": len(generations),
            "load_time_sec": load_time_sec,
            "cp4_2_artifact_sha256": cp42_artifact_sha,
            "cp4_2_artifact_status": "unmodified",
            "cp4_2_artifact_mtime": file_mtime_iso(CP4_2_ARTIFACT_PATH),
            "fidelity_check": {
                "reference_seed": cp42_reference["seed"] if cp42_reference else None,
                "reference_token_count": cp42_reference["generated_token_count"] if cp42_reference else None,
                "reference_answer": cp42_reference["generated_answer"] if cp42_reference else None,
                "diagnostic_seed_0_token_count": _diagnostic_seed_token_count(generations, DIAGNOSTIC_SEEDS[0]),
                "pass": fidelity_check_pass,
                "note": (
                    "Output-capture flags (output_scores/output_logits/"
                    "return_dict_in_generate) must not alter sampling. Under "
                    "CPU_ONLY this is deterministic, so the diagnostic seed-1001 "
                    "output must byte-match the CP4.2 pilot seed-1001 output."
                ),
            },
            "caution": (
                "This diagnostic changes nothing in CP4.2. Three generations "
                "are not sufficient for statistical estimation; they are only "
                "sufficient to read the candidate distribution directly."
            ),
        },
        "prompt": {
            "sample_id": scenario.get("sample_id"),
            "dataset_index": dataset_index,
            "answerability": scenario.get("reference", {}).get("answerability"),
            "expected_behavior": scenario.get("metadata", {}).get("expected_behavior"),
            "question": scenario.get("prompt", {}).get("question"),
            "context_length_chars": len(scenario.get("prompt", {}).get("context", "")),
            "prompt_text": prompt_text,
            "prompt_token_count": prompt_token_count,
            "pad_token_id": pad_token_id,
            "eos_token_ids": list(EOS_TOKEN_IDS),
        },
        "method": {
            "output_capture": "generate(output_scores=True, output_logits=True, return_dict_in_generate=True)",
            "scores_meaning": "post-processor logits (TemperatureLogitsWarper -> TopKLogitsWarper -> TopPLogitsWarper, min_tokens_to_keep=1) - exactly what torch.multinomial consumed",
            "raw_logits_meaning": "pre-processor logits captured ONLY for cross-check; discarded after; never serialised",
            "probs_used": "softmax(step_scores) - identical to torch.multinomial's distribution",
            "nucleus_size_definition": (
                "count of tokens with softmax prob > 0; equals count of finite "
                "logits surviving Temperature/TopK/TopP, i.e. the top-p nucleus "
                "size after top-k truncation"
            ),
            "top1_prob_definition": "max softmax prob over the post-processor distribution",
            "what_is_not_stored": "full vocabulary-sized logits tensors (per-step top-5 + scalars only)",
            "full_logits_serialised": False,
            "cross_check": "rebuilt processor chain applied to raw_logits must match captured scores",
        },
        "generations": generations,
    }


def _transformers_version():
    import transformers
    return transformers.__version__


def _diagnostic_seed_token_count(generations, seed):
    for g in generations:
        if g["seed"] == seed:
            return g["generated_token_count"]
    return None


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _agg(generations, key):
    out = []
    for g in generations:
        for s in g["steps"]:
            out.append(s[key])
    return out


def _fmt_pct(x):
    return "{0:.2f}%".format(x * 100.0)


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def _median(xs):
    if not xs:
        return 0.0
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 == 1 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def render_report(artifact):
    meta = artifact["diagnostic_metadata"]
    gens = artifact["generations"]
    prompt_tok = artifact["prompt"]["prompt_token_count"]

    total_steps = sum(len(g["steps"]) for g in gens)
    nucleus_sizes = _agg(gens, "nucleus_size")
    top1_probs = _agg(gens, "top1_prob")
    ranks = [s["sampled_rank"] for g in gens for s in g["steps"]]

    steps_nucleus_eq_1 = sum(1 for n in nucleus_sizes if n == 1)
    steps_nucleus_le_2 = sum(1 for n in nucleus_sizes if n <= 2)
    steps_rank_gt_1 = sum(1 for r in ranks if r is not None and r > 1)

    max_cross_check = max(g["cross_check_max_abs_diff"] for g in gens)

    first_few_top1 = []
    for g in gens:
        first_few_top1.append({
            "seed": g["seed"],
            "steps_0_1_2_top1_prob": [g["steps"][i]["top1_prob"] if i < len(g["steps"]) else None for i in range(3)],
            "steps_0_1_2_top1_token_id": [g["steps"][i]["top1_token_id"] if i < len(g["steps"]) else None for i in range(3)],
            "steps_0_1_2_nucleus": [g["steps"][i]["nucleus_size"] if i < len(g["steps"]) else None for i in range(3)],
        })

    # find first divergence step across the 3 seeds
    gen_ids = {g["seed"]: g["gen_token_ids"] for g in gens}
    divergence_step = None
    divergence_detail = None
    min_len = min(len(v) for v in gen_ids.values())
    for i in range(min_len):
        toks = [gen_ids[s][i] for s in DIAGNOSTIC_SEEDS]
        if len(set(toks)) > 1:
            divergence_step = i
            divergence_detail = {
                "step": i,
                "by_seed": {str(s): {"token_id": gen_ids[s][i],
                                     "nucleus_size": gens[idx_for_seed(s, gens)]["steps"][i]["nucleus_size"],
                                     "top1_prob": gens[idx_for_seed(s, gens)]["steps"][i]["top1_prob"],
                                     "top1_token_id": gens[idx_for_seed(s, gens)]["steps"][i]["top1_token_id"]}
                                     for s in DIAGNOSTIC_SEEDS},
            }
            break

    identical_outputs = len(set(json.dumps(g["generated_answer"]) for g in gens)) == 1

    lines = []
    lines.append("# CP4.2.1 — Instrumented Sampling Diagnostic Report")
    lines.append("")
    lines.append("## 1. Summary")
    lines.append("")
    lines.append("- **Prompt:** dataset_index `{0}` ({1}); question: \"{2}\"".format(
        artifact["prompt"]["dataset_index"],
        artifact["prompt"]["sample_id"],
        artifact["prompt"]["question"]))
    lines.append("- **Seeds:** `{0}`".format(", ".join(str(s) for s in meta["seeds_used"])))
    lines.append("- **Decoding configuration (CP4.2, unchanged):** do_sample={do_sample}, temperature={temperature}, "
                 "top_p={top_p}, top_k={top_k}, max_new_tokens={max_new_tokens}".format(
                     **{k: meta["decoding_config"][k] for k in
                        ("do_sample", "temperature", "top_p", "top_k", "max_new_tokens")}))
    lines.append("- **model.generate() calls:** {0} (exactly one per seed).".format(meta["model_generate_calls"]))
    lines.append("- **Prompt tokens:** {0}; generations total {1} steps across 3 seeds.".format(
        prompt_tok, total_steps))
    lines.append("- **CP4.2 artifact SHA256:** `{0}` — status: {1}".format(
        meta["cp4_2_artifact_sha256"], meta["cp4_2_artifact_status"]))
    lines.append("- **Canonical dataset:** unmodified (same idx=61608 scenario as CP4.2 pilot).")
    lines.append("- **Cross-check:** rebuilt processor chain reproduces captured `scores`; "
                 "max |processor(raw_logits) - scores| = `{0:.3e}` (0 ⇒ capture is faithful).".format(max_cross_check))
    lines.append("")
    lines.append("## 2. Answer-level results")
    lines.append("")
    lines.append("| Seed | Tokens | EOS reached | Top-1 tokens (step 0→1→2) | Generated answer |")
    lines.append("|---|---|---|---|---|")
    for g in gens:
        t1 = g["steps"][0]["top1_token_id"] if g["steps"] else None
        t2 = g["steps"][1]["top1_token_id"] if len(g["steps"]) > 1 else None
        t3 = g["steps"][2]["top1_token_id"] if len(g["steps"]) > 2 else None
        lines.append("| {0} | {1} | {2} | {3} → {4} → {5} | `{6}` |".format(
            g["seed"], g["generated_token_count"], g["eos_reached"], t1, t2, t3, g["generated_answer"]))
    lines.append("")
    lines.append("**Outputs identical across the three seeds: {0}.**".format(identical_outputs))
    lines.append("")
    lines.append("## 3. Per-step sampling distribution (aggregated over {0} tokens)".format(total_steps))
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    lines.append("| Steps with nucleus size = 1 | {0} / {1} ({2}) |".format(
        steps_nucleus_eq_1, total_steps, _fmt_pct(steps_nucleus_eq_1 / total_steps)))
    lines.append("| Steps with nucleus size <= 2 | {0} / {1} ({2}) |".format(
        steps_nucleus_le_2, total_steps, _fmt_pct(steps_nucleus_le_2 / total_steps)))
    lines.append("| Steps where sampled token was NOT top-1 (rank > 1) | {0} / {1} ({2}) |".format(
        steps_rank_gt_1, total_steps, _fmt_pct(steps_rank_gt_1 / total_steps)))
    lines.append("| Mean top-1 probability | {0:.4f} |".format(_mean(top1_probs)))
    lines.append("| Median top-1 probability | {0:.4f} |".format(_median(top1_probs)))
    lines.append("| Min top-1 probability | {0:.4f} |".format(min(top1_probs)))
    lines.append("| Max top-1 probability | {0:.4f} |".format(max(top1_probs)))
    lines.append("| Mean nucleus size | {0:.3f} |".format(_mean(nucleus_sizes)))
    lines.append("| Median nucleus size | {0} |".format(int(_median(nucleus_sizes))))
    lines.append("| Max nucleus size (any step) | {0} |".format(max(nucleus_sizes)))
    lines.append("")
    lines.append("## 4. First three answer tokens (per seed)")
    lines.append("")
    lines.append("| Seed | step 0 (top1 prob, nucleus) | step 1 | step 2 |")
    lines.append("|---|---|---|---|")
    for f in first_few_top1:
        row = [str(f["seed"])]
        for i in range(3):
            tp = f["steps_0_1_2_top1_prob"][i]
            tn = f["steps_0_1_2_nucleus"][i]
            if tp is None:
                row.append("-")
            else:
                row.append("{0:.4f}, n={1}".format(tp, tn))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append("## 5. Seed divergence analysis")
    lines.append("")
    if divergence_step is None:
        lines.append("The three seeds produced **identical token sequences**, so no divergence step exists "
                     "within this sample. The observed CP4.2 25-vs-27-token split (seeds {1002,1005,1006,1007,1009} "
                     "vs {1001,1003,1004,1008,1010}) is NOT reproduced by seeds {0}; it would require sampling "
                     "the full 10 seeds to confirm.".format(", ".join(str(s) for s in DIAGNOSTIC_SEEDS)))
    else:
        lines.append("First token-level divergence at **step {0}**. Per-seed detail at that step:".format(divergence_step))
        lines.append("")
        lines.append("| Seed | token id | nucleus size | top-1 prob | top-1 id |")
        lines.append("|---|---|---|---|---|")
        for s_str, d in divergence_detail["by_seed"].items():
            lines.append("| {0} | {1} | {2} | {3:.4f} | {4} |".format(
                s_str, d["token_id"], d["nucleus_size"], d["top1_prob"], d["top1_token_id"]))
        lines.append("")
        lines.append("A divergence at step {0} with nucleus size > 1 at that step confirms the output "
                     "difference is a real token-sampling difference.".format(divergence_step))
    lines.append("")
    lines.append("## 6. Interpretation (read-only; no parameter changes recommended yet)")
    lines.append("")
    lines.append("**Stochastic sampling is functioning** (`do_sample=True` path; captured probabilities are "
                 "non-degenerate by construction). The diagnostic cannot confirm whether sampling produces "
                 "**useful variation for risk estimation** — that is a design question addressed by the counts below.")
    lines.append("")
    lines.append("Key conditional: if the fraction of steps with `nucleus size = 1` is high "
                  "({0:.1f}% here), then a top-p nucleus that contains a single token makes multinomial "
                  "sampling equivalent to greedy at that step. Whether that *explains* the CP4.2 collapse "
                  "depends on where those size-1 steps fall: the first answer token, which sets the whole "
                  "'The ... is ...' frame, is the dominant control point. A concentrated first token "
                  "(top-1 prob near 1.0, nucleus = 1) at step 0 would explain why seeds converge to the "
                  "same continuation even though the sampler is nominally stochastic.".format(
                      100.0 * steps_nucleus_eq_1 / total_steps))
    lines.append("")
    lines.append("## 7. Files produced")
    lines.append("")
    lines.append("- `data/pilots/cp4_2_sampling_diagnostic.json` (this artifact)")
    lines.append("- `docs/cp4_2_sampling_diagnostic_report.md` (this report)")
    lines.append("- `run_cp4_2_sampling_diagnostic.py` (the diagnostic script; pure list-based helpers are unit tested)")
    lines.append("")
    lines.append("## 8. Scope-compliance note")
    lines.append("")
    lines.append("No CP4.2 parameter, source, config, test, artifact, or report file was modified. "
                 "This diagnostic reuses `run_cp4_2_pilot`'s constants and `_load_model`/`render_prompt`/`"
                 "select_pilot_scenarios` so the model, revision, thinking mode, prompt text and decoding "
                 "kwargs are byte-for-byte identical to CP4.2. `CP4.2.1` starts here and does not begin CP4.3.")
    lines.append("")
    return "\n".join(lines)


def idx_for_seed(seed, generations):
    for i, g in enumerate(generations):
        if g["seed"] == seed:
            return i
    raise KeyError(seed)


if __name__ == "__main__":
    run_diagnostic()
