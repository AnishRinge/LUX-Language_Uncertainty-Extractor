"""CP4.5-A live Gemini evaluator smoke test.

Purpose: verify operational readiness of the real Gemini evaluator end-to-end
on a tiny deterministic sample (one generation per representative prompt from
the existing CP4.2.5 artifact).

This is NOT evaluator validation, NOT human validation, NOT target construction.

HARD CONSTRAINT (request inputs):
    The Gemini evaluator receives ONLY question, supplied SQuAD context, and the
    generated Qwen3 answer. build_messages() is called with exactly those three
    arguments; no seed, model identity, logits, hidden states, temperature,
    generation index, or predictor features are sent to the Gemini API.

If GEMINI_API_KEY is not present in the environment, this script does NOT run
evaluations, does NOT fake results, and does NOT write any artifact. It exits
non-zero with a clear message so a missing key is never silently bypassed.
"""

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Make `src` importable when run as a script.
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.evaluation.gemini_evaluator import (  # noqa: E402
    API_KEY_ENV_VAR,
    EvaluationRequest,
    GeminiEvaluator,
)
from src.evaluation.schemas import (  # noqa: E402
    EvaluationRecord,
    SCHEMA_VERSION,
)
from src.evaluation.retry import RetryPolicy  # noqa: E402

ARTIFACT = Path("data/pilots/cp4_2_5_representative_validation.json")
SMOKE_OUTPUTS = {
    "artifact": Path("data/pilots/cp4_5_a_gemini_smoke_test.json"),
    "report": Path("docs/cp4_5_a_gemini_smoke_test_report.md"),
}

TARGET_DATASET_INDEXES = [19919, 61608, 108548, 31073, 69929, 108653]
SEED = 1001

# ---------------------------------------------------------------------------
# CP4.5-A smoke-test execution protocol (quota-safe for the free tier).
#
# max_retries=0 applies ONLY to this smoke-test invocation. It overrides the
# evaluator's default retry policy for this run only (the evaluator's default
# RetryPolicy() used by normal runs is unchanged). One attempt per case plus
# ~15s spacing keeps us within the free-tier limit of 5 requests/minute while
# each case still gets exactly one real Gemini call.
# ---------------------------------------------------------------------------
SMOKE_MAX_RETRIES = 0
SMOKE_CALL_SPACING_SECONDS = 15
SMOKE_RETRY_POLICY = RetryPolicy(
    max_retries=SMOKE_MAX_RETRIES, sleep=lambda s: None
)


def _extract_user_block(prompt_text: str) -> str:
    """Return the user-message content from a Qwen-formatted prompt_text."""
    start = prompt_text.find("<|im_start|>user")
    if start == -1:
        raise ValueError("could not locate user block in prompt_text")
    remainder = prompt_text[start + len("<|im_start|>user"):]
    end = remainder.find("<|im_end|>")
    if end == -1:
        raise ValueError("could not locate end of user block in prompt_text")
    return remainder[:end].strip()


def _split_question_context(user_block: str):
    """Split a SQuAD user block into (context, question).

    Convention in this artifact: the supplied context precedes the question and
    they are separated by a blank line; the question terminates with '?'.
    """
    parts = user_block.rsplit("\n\n", 1)
    if len(parts) == 2 and parts[1].strip().endswith("?"):
        return parts[0].strip(), parts[1].strip()
    # Fall back: treat the whole block as context with no separable question.
    return user_block, ""


def select_samples(artifact):
    """Deterministically select one generation per target dataset_index."""
    gens = artifact["generations"]
    selected = {}
    for g in gens:
        di = g.get("dataset_index")
        if di in TARGET_DATASET_INDEXES and di not in selected and g.get("seed") == SEED:
            selected[di] = g
    if len(selected) != len(TARGET_DATASET_INDEXES):
        missing = set(TARGET_DATASET_INDEXES) - set(selected.keys())
        raise RuntimeError("could not locate all six targets; missing: {0}".format(missing))
    return [selected[i] for i in TARGET_DATASET_INDEXES]


def build_request(gen):
    user_block = _extract_user_block(gen["prompt_text"])
    context, question = _split_question_context(user_block)
    if not question:
        raise RuntimeError(
            "could not recover a question for dataset_index={0}".format(gen.get("dataset_index"))
        )
    return EvaluationRequest(
        # --- ONLY these three reach the Gemini evaluator payload:
        question=question,
        context=context,
        generated_answer=gen["generated_answer"],
        # --- provenance only (NOT sent to Gemini):
        source_generation_id="squad_v2:{0}:seed{1}".format(gen["dataset_index"], SEED),
        dataset_id="squad_v2",
        sample_id=str(gen.get("sample_id", "")),
        generation_index=0,
        seed=SEED,
        model_name="",
        model_revision="",
        answerability=gen.get("answerability", ""),
    )


def _load_env_file() -> None:
    """Best-effort: load ``GEMINI_API_KEY`` from a ``.env`` file into os.environ.

    The user supplies the key via a ``.env`` file (e.g. ``.venv/.env``). This
    materializes it as the ``GEMINI_API_KEY`` environment variable so the
    evaluator's existing env-var contract holds. The key is NEVER read from disk
    back to the caller, NEVER printed, and only ``GEMINI_API_KEY`` is honored.
    """
    if os.environ.get(API_KEY_ENV_VAR):
        return
    candidates = [Path(".env"), Path(".venv/.env")]
    candidates += sorted(p for p in Path(".").glob("**/.env") if p != Path(".env"))
    for path in candidates:
        if not path.is_file():
            continue
        try:
            for raw in path.read_text(encoding="utf-8").splitlines():
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key == API_KEY_ENV_VAR:
                    os.environ[API_KEY_ENV_VAR] = value
                    return
        except OSError:
            continue


def run():
    _load_env_file()
    # Gate: never proceed without a key, never fake.
    key = os.environ.get(API_KEY_ENV_VAR)
    if not key:
        print(
            "SMOKE TEST ABORTED: GEMINI_API_KEY is not set in the environment. "
            "No evaluations were performed and no artifact was written. "
            "Provide a valid {0} to run the live smoke test.".format(API_KEY_ENV_VAR),
            file=sys.stderr,
        )
        sys.exit(2)

    if not ARTIFACT.exists():
        raise SystemExit("missing source artifact: {0}".format(ARTIFACT))

    with open(ARTIFACT, "r", encoding="utf-8") as fh:
        artifact = json.load(fh)

    selected = select_samples(artifact)

    # Real client: no injected client, no SDK at import time (lazy import in
    # GeminiEvaluator._build_client). The configured model from prompt.EVALUATOR_MODEL
    # is used. The zero-retry policy here is the smoke-test protocol only; it
    # does not alter the evaluator's default retry policy.
    evaluator = GeminiEvaluator(retry_policy=SMOKE_RETRY_POLICY)
    model_id = evaluator.model

    results = []
    for index, gen in enumerate(selected):
        if index > 0:
            # Space calls to respect the free-tier 5 requests/minute limit.
            time.sleep(SMOKE_CALL_SPACING_SECONDS)
        request = build_request(gen)
        record = evaluator.evaluate(request)
        record.validate()
        results.append(
            {
                "dataset_index": gen["dataset_index"],
                "sample_id": gen.get("sample_id"),
                "evaluation_status": record.evaluation_status.value,
                "generation_label": (
                    record.generation_label.value if record.generation_label is not None else None
                ),
                "claim_count": len(record.claims),
                "needs_human_review": record.needs_human_review,
                "retry_count": record.retry_count,
                "error_type": record.error_type,
                "error_message": record.error_message,
                "evaluator_model": record.evaluator_model,
                "prompt_version": record.prompt_version,
                "schema_version": record.schema_version,
            }
        )

    summary = {
        "success": sum(
            1 for r in results if r["evaluation_status"] == "SUCCESS"
        ),
        "manual_review": sum(
            1 for r in results if r["evaluation_status"] == "MANUAL_REVIEW"
        ),
        "failed": sum(1 for r in results if r["evaluation_status"] == "FAILED"),
        "retries": sum(r["retry_count"] for r in results),
        "label_distribution": {
            lbl: sum(1 for r in results if r["generation_label"] == lbl)
            for lbl in ("RELIABLE", "UNRELIABLE", "INADEQUATE", None)
        },
    }

    artifact_obj = {
        "run_metadata": {
            "test": "CP4.5-A live Gemini evaluator smoke test",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "evaluator_model": model_id,
            "source_artifact": str(ARTIFACT),
            "selected_dataset_indexes": TARGET_DATASET_INDEXES,
            "select_seed": SEED,
            "schema_version": SCHEMA_VERSION,
            "smoke_retry_max_retries": SMOKE_MAX_RETRIES,
            "smoke_call_spacing_seconds": SMOKE_CALL_SPACING_SECONDS,
        },
        "constraints_verified": {
            "request_inputs": ["question", "supplied context", "generated answer"],
            "forbidden_inputs_not_sent": [
                "model identity",
                "revision",
                "seed",
                "temperature",
                "top_p",
                "top_k",
                "max_new_tokens",
                "generation index",
                "logits",
                "hidden states",
                "token probabilities",
                "evaluator metadata revealing target model",
            ],
        },
        "records": results,
        "summary": summary,
    }

    SMOKE_OUTPUTS["artifact"].parent.mkdir(parents=True, exist_ok=True)
    with open(SMOKE_OUTPUTS["artifact"], "w", encoding="utf-8") as fh:
        json.dump(artifact_obj, fh, indent=2)
        fh.write("\n")

    # Re-load check: the persisted artifact must parse again.
    reloaded = json.loads(SMOKE_OUTPUTS["artifact"].read_text(encoding="utf-8"))
    assert len(reloaded["records"]) == len(results)

    report_lines = [
        "# CP4.5-A — Live Gemini Evaluator Smoke Test Report",
        "",
        "## Status",
        "Live smoke test executed (6 evaluations).",
        "",
        "## Run metadata",
        "- evaluator model: `{0}`".format(model_id),
        "- source artifact: `{0}`".format(ARTIFACT),
        "- selected dataset_index values: {0}".format(TARGET_DATASET_INDEXES),
        "- generated at (UTC): {0}".format(artifact_obj["run_metadata"]["generated_at_utc"]),
        "",
        "## Request-input constraint verification",
        "The evaluator was built so `build_messages` is called with ONLY `question`,",
        "`context`, and `generated_answer`. No Qwen3 model identity, seed, decoding",
        "config, logits, hidden states, or predictor features are sent to the Gemini API.",
        "",
        "## Summary",
        "- successful evaluations: {0}".format(summary["success"]),
        "- manual-review evaluations: {0}".format(summary["manual_review"]),
        "- failed evaluations: {0}".format(summary["failed"]),
        "- total retries: {0}".format(summary["retries"]),
        "- generation-label distribution: {0}".format(summary["label_distribution"]),
        "",
        "## Records",
        "| dataset_index | status | label | claims | review | retries |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        report_lines.append(
            "| {0} | {1} | {2} | {3} | {4} | {5} |".format(
                r["dataset_index"],
                r["evaluation_status"],
                r["generation_label"],
                r["claim_count"],
                r["needs_human_review"],
                r["retry_count"],
            )
        )
    report_lines.append("")
    report_lines.append("## Artifact")
    report_lines.append("`{0}`".format(SMOKE_OUTPUTS["artifact"]))
    SMOKE_OUTPUTS["report"].parent.mkdir(parents=True, exist_ok=True)
    SMOKE_OUTPUTS["report"].write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print("SMOKE TEST COMPLETE: 6 evaluations.")
    print(json.dumps(summary, indent=2))
    print("artifact: {0}".format(SMOKE_OUTPUTS["artifact"]))
    print("report:   {0}".format(SMOKE_OUTPUTS["report"]))


if __name__ == "__main__":
    run()
