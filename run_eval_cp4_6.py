#!/usr/bin/env python
"""Run or resume isolated Gemini 2.5 Flash-Lite CP4.6.0 evaluations."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_eval_cp4_5 as runner
from src.evaluation.prompt_cp4_6 import (
    CHANGE_REASON,
    EVALUATOR_MODEL,
    PROMPT_VERSION,
    build_messages,
    response_schema,
)
from src.evaluation.retry import RetryPolicy

VERSION_DIR = ROOT / "data" / "production" / "cp4_6_0_gemini_2_5_flash_lite"
EVALUATIONS_DIR = VERSION_DIR / "evaluations"
RUN_LOG_PATH = VERSION_DIR / "evaluation_run.json"
TARGETS_PATH = VERSION_DIR / "targets.json"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--spacing", type=float, default=1.0)
    evaluate.add_argument("--max-seconds", type=int, default=9000)
    evaluate.add_argument("--max-evaluations", type=int)
    evaluate.add_argument(
        "--single-attempt",
        action="store_true",
        help="Use only the primary credential and disable retries for one controlled request",
    )
    target = subparsers.add_parser("build-targets")
    target.add_argument("--target-path", type=Path, default=TARGETS_PATH)
    args = parser.parse_args(argv)

    if args.command == "evaluate":
        import run_production

        run_production._load_env_file()
        result = runner.run_evaluation(
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
            evaluator_options=(
                {
                    "max_credential_attempts": 1,
                    "retry_policy": RetryPolicy(max_retries=0),
                }
                if args.single_attempt
                else None
            ),
        )
        return 0 if result["stop_reason"] == "completed" else 2

    result = runner.build_targets(
        evaluations_dir=EVALUATIONS_DIR,
        targets_path=args.target_path,
        evaluator_model=EVALUATOR_MODEL,
        prompt_version=PROMPT_VERSION,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
