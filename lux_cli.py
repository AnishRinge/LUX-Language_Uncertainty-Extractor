"""Interactive CLI for the LUX baseline inference interface.

Usage:
    python lux_cli.py                      # interactive: type question and context
    python lux_cli.py --text "..."        # one-shot rendered prompt text
    python lux_cli.py --median            # use the median artifact (returns 0.0)
    python lux_cli.py --verify 0.2 --escalate 0.5 --allow-uncalibrated

The returned risk is a cohort constant, not a prompt-specific estimate.
No target model or evaluator is loaded.
"""
import argparse
import sys

from src.predictor.inference import (
    Artifact, GENERATION_CONFIG, MODEL, REVISION, Policy, predict_risk,
)

MEAN_ARTIFACT = "docs/engineering/baseline_mean_v2.json"
MEDIAN_ARTIFACT = "docs/engineering/baseline_median_v2.json"


def read_prompt(args):
    if args.text is not None:
        return args.text
    print("Enter the question (empty line to finish):")
    question = _read_block()
    print("Enter the context/passage (empty line to finish):")
    context = _read_block()
    if not question.strip() and not context.strip():
        raise SystemExit("No prompt entered.")
    return {"question": question.strip(), "context": context.strip()}


def _read_block():
    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line == "":
            break
        lines.append(line)
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description="LUX baseline risk CLI")
    parser.add_argument("--text", default=None, help="rendered prompt text")
    parser.add_argument("--median", action="store_true", help="use median artifact")
    parser.add_argument("--artifact", default=None, help="path to a baseline JSON artifact")
    parser.add_argument("--verify", type=float, default=None)
    parser.add_argument("--escalate", type=float, default=None)
    parser.add_argument("--allow-uncalibrated", action="store_true")
    args = parser.parse_args(argv)

    artifact_path = args.artifact or (MEDIAN_ARTIFACT if args.median else MEAN_ARTIFACT)
    try:
        artifact = Artifact.load(artifact_path)
    except Exception as exc:  # noqa: BLE001 - surface any load failure to the user
        print(f"Failed to load artifact {artifact_path}: {exc}", file=sys.stderr)
        return 1

    try:
        prompt = read_prompt(args)
    except SystemExit as exc:
        print(str(exc), file=sys.stderr)
        return 1

    try:
        prediction = predict_risk(
            prompt, MODEL, REVISION, dict(GENERATION_CONFIG), artifact=artifact,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Prediction rejected: {exc}", file=sys.stderr)
        return 1

    policy = Policy(verify_at=args.verify, escalate_at=args.escalate,
                    allow_uncalibrated=args.allow_uncalibrated)
    action = policy.route(prediction)

    print("\n--- LUX baseline result ---")
    print(f"artifact        : {artifact_path}")
    print(f"baseline        : {artifact.statistic} = {artifact.value}")
    print(f"estimated_risk  : {prediction.estimated_risk}")
    print(f"calibrated      : {prediction.calibrated}")
    print(f"uncertainty     : {prediction.uncertainty}")
    print(f"action          : {action}")
    print("warnings:")
    for w in prediction.warnings:
        print(f"  - {w}")
    print("---------------------------")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())