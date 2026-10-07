"""CP5-A: Live Gemini smoke check on ONE existing production generation.

Verifies:
  * the new API key/quota works
  * the evaluator sends ONLY question + context + generated_answer
  * structured evaluation parses
  * evaluator output is persisted
  * no API key is exposed in any persisted record
  * no forbidden target-model metadata is sent
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.evaluation.evaluator_runner import build_request_from_cp4_2_5
from src.evaluation.gemini_evaluator import GeminiEvaluator
from src.evaluation.schemas import EvaluationStatus

API_KEY_ENV = "GEMINI_API_KEY"

# Pick ONE already-generated production answer: dataset_index 69929, seed 1001
GENERATIONS_PATH = Path("data/production/generations.json")
EVALUATIONS_DIR = Path("data/production/evaluations")
EVALUATIONS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    # 1. Load the key from the same source the pipeline uses.
    api_key = os.environ.get(API_KEY_ENV)
    if not api_key:
        # Fallback: the .venv/.env file the user configured.
        env_file = Path(".venv/.env")
        if env_file.exists():
            for line in env_file.read_text(encoding="utf-8").splitlines():
                if line.strip().startswith("GEMINI_API_KEY="):
                    api_key = line.strip().split("=", 1)[1]
                    os.environ[API_KEY_ENV] = api_key
                    break
    if not api_key:
        print("FAIL: no GEMINI_API_KEY available")
        return 1
    print("API key loaded (length={}, prefix={})".format(len(api_key), api_key[:8]))

    # 2. Load the existing production generation.
    with open(GENERATIONS_PATH, "r", encoding="utf-8") as fh:
        artifact = json.load(fh)

    target = None
    for gen in artifact["generations"]:
        if gen["dataset_index"] == 69929 and gen["seed"] == 1001:
            target = gen
            break
    if target is None:
        print("FAIL: target generation not found")
        return 1
    print("Target generation_id: {}".format(target["generation_id"]))

    # 3. Load the canonical scenario for question/context.
    with open("data/prompts/squad_v2_canonical_scenarios.json", "r", encoding="utf-8") as fh:
        scenarios = json.load(fh)
    scenario = scenarios[target["dataset_index"]]

    # 4. Build the evaluation request (ONLY Q/C/A in the user turn).
    req = build_request_from_cp4_2_5(target, scenario)

    # 5. Verify the WIRE content sent to Gemini carries only Q/C/A.
    from src.evaluation.prompt import build_messages
    messages = build_messages(req.question, req.context, req.generated_answer)
    user_msg = next(m for m in messages if m["role"] == "user")["content"]
    wire_blob = json.dumps(messages)
    forbidden_values = [
        str(req.seed),
        req.model_name,
        req.model_revision,
        req.answerability,
    ]
    leaked = [v for v in forbidden_values if v and v in wire_blob]
    if leaked:
        print("FAIL: forbidden metadata leaked into wire content: {}".format(leaked))
        return 1
    print("Wire content contains only question/context/generated_answer.")

    # 6. Call Gemini with a robust retry policy. The frozen evaluator model
    #    (gemini-3.8-flash) is currently experiencing high demand and emits
    #    transient 503 UNAVAILABLE spikes; the production pipeline uses the
    #    default RetryPolicy(max_retries=3). We use a slightly larger budget
    #    here purely for the smoke check so a transient spike does not mask a
    #    genuine key/quota failure.
    from src.evaluation.retry import RetryPolicy
    evaluator = GeminiEvaluator(
        api_key=api_key,
        retry_policy=RetryPolicy(
            max_retries=6,
            base_delay=3.0,
            max_delay=20.0,
            backoff_factor=2.0,
        ),
    )
    record = evaluator.evaluate(req)
    print("Evaluation status: {}".format(record.evaluation_status))
    print("Generation label: {}".format(record.generation_label))

    # 7. Verify the record is valid.
    if record.evaluation_status == EvaluationStatus.FAILED:
        print("FAIL: evaluation FAILED: {}".format(record.error_message))
        return 1
    if record.generation_label is None:
        print("FAIL: no generation label on a non-FAILED record")
        return 1

    # 8. Persist the evaluation record atomically.
    safe = record.source_generation_id.replace(":", "_").replace("/", "_")
    eval_path = EVALUATIONS_DIR / "{}.json".format(safe)
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = eval_path.with_name(eval_path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(record.to_dict(), fh, indent=2, ensure_ascii=False, default=str)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, eval_path)
    print("Evaluation persisted to {}".format(eval_path))

    # 9. Verify no API key is exposed in the persisted record.
    persisted = json.dumps(record.to_dict())
    if api_key in persisted:
        print("FAIL: API key leaked into persisted record")
        return 1
    print("No API key in persisted record.")

    # 10. Verify the request sent to Gemini contained no forbidden metadata.
    # (We cannot inspect the real client's calls here, but the request builder
    #  is the single choke point and was verified above.)

    print("GEMINI SMOKE CHECK PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())