#!/usr/bin/env python
"""CLI entry point for comparing hallucination-risk predictors.

Trains and evaluates three baselines with a unified comparison interface:
  - PromptOnly: Ridge regression on 5 prompt features (no hidden states)
  - HiddenStateOnly: Ridge regression on PCA(50) of last-layer hidden states
  - Fusion: Ridge regression on concatenated [prompt features + PCA hidden states]

All three share the same train/test split and empirical-risk targets when
sufficient data is available.

Usage:
    python run_predictor.py compare
    python run_predictor.py compare --test-size 0.25 --random-state 42
    python run_predictor.py report
"""

import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.predictor.baselines import (
    DEFAULT_FEATURES_PATH,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_RANDOM_STATE,
    DEFAULT_TARGETS_PATH,
    DEFAULT_HIDDEN_STATES_DIR,
    AlignmentError,
    PromptOnlyPredictor,
    TargetRow,
    _compute_dataset_stats,
    _compute_dataset_stats_with_hidden,
    load_predictor_targets,
    prompt_only_train_test_split,
)
from src.predictor.hidden_only import (
    HIDDEN_STATE_STRATEGY,
    HiddenStateOnlyPredictor,
    align_hidden_states_with_targets,
    extract_hidden_representation,
    load_hidden_states,
    run_hidden_only_baseline,
)
from src.predictor.fusion import (
    FusionPredictor,
    align_feature_hidden_fusion,
    run_fusion_baseline,
)
from src.predictor.baselines import (
    train_test_split_by_prompt,
    MIN_TRAIN_PROMPTS,
    MIN_TEST_PROMPTS,
)
from src.predictor.baselines import ALL_FEATURE_NAMES


def _print_header(title: str) -> None:
    print("=" * 72)
    print(title)
    print("=" * 72)


def _print_section(title: str) -> None:
    print("\n" + "-" * 72)
    print(title)
    print("-" * 72)


def _bootstrap_metric_intervals(
    targets: List[float],
    predictions: List[float],
    baseline_predictions: Optional[List[float]] = None,
    *,
    random_state: int = DEFAULT_RANDOM_STATE,
    resamples: int = 2000,
) -> Dict[str, Any]:
    """Percentile bootstrap intervals over paired held-out prompt predictions."""
    y_true = np.asarray(targets, dtype=float)
    y_pred = np.asarray(predictions, dtype=float)
    if y_true.shape != y_pred.shape or y_true.ndim != 1:
        raise ValueError("Bootstrap targets and predictions must be aligned vectors")
    if len(y_true) < 2:
        return {"method": "paired prompt bootstrap", "n_test": len(y_true)}
    rng = np.random.default_rng(random_state)
    indices = rng.integers(0, len(y_true), size=(resamples, len(y_true)))
    errors = y_pred[indices] - y_true[indices]
    mae = np.mean(np.abs(errors), axis=1)
    rmse = np.sqrt(np.mean(np.square(errors), axis=1))
    result = {
        "method": "paired prompt bootstrap percentile interval",
        "resamples": resamples,
        "n_test": len(y_true),
        "mae_95_percentile_interval": [
            float(v) for v in np.quantile(mae, [0.025, 0.975])
        ],
        "rmse_95_percentile_interval": [
            float(v) for v in np.quantile(rmse, [0.025, 0.975])
        ],
    }
    if baseline_predictions is not None:
        baseline = np.asarray(baseline_predictions, dtype=float)
        if baseline.shape != y_true.shape:
            raise ValueError("Bootstrap baseline predictions must align with targets")
        baseline_mae = np.mean(
            np.abs(baseline[indices] - y_true[indices]), axis=1
        )
        improvement = baseline_mae - mae
        result["mae_improvement_over_baseline_95_percentile_interval"] = [
            float(v) for v in np.quantile(improvement, [0.025, 0.975])
        ]
    return result


def _target_distribution(rows, prompt_ids: List[str]) -> Dict[str, Any]:
    selected = [row.risk for row in rows if row.prompt_id in set(prompt_ids)]
    counts = Counter(f"{risk:.1f}" for risk in selected)
    return {
        "n_prompts": len(selected),
        "risk_counts": dict(sorted(counts.items())),
        "mean": float(np.mean(selected)) if selected else None,
        "min": float(min(selected)) if selected else None,
        "max": float(max(selected)) if selected else None,
    }


def run_comparison(
    features_path: Path = DEFAULT_FEATURES_PATH,
    targets_path: Path = DEFAULT_TARGETS_PATH,
    hidden_states_dir: Path = DEFAULT_HIDDEN_STATES_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    test_size: float = 0.3,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> Dict[str, Any]:
    """Run all three baselines with a shared train/test split.

    When sufficient data is available (≥5 valid targets), all three predictors
    are evaluated on the SAME train/test prompt partition to enable a fair
    comparison. Otherwise, each baseline reports its individual status.
    """
    _print_header("LUX Hallucination-Risk Predictor Comparison")
    print("Features: {0}".format(features_path))
    print("Targets:  {0}".format(targets_path))
    print("Hidden:   {0}".format(hidden_states_dir))
    print("Output:   {0}".format(output_dir))

    # --- Step 1: Shared dataset statistics ---
    _print_section("Step 1: Dataset Statistics")
    stats = _compute_dataset_stats_with_hidden(
        features_path, targets_path, hidden_states_dir
    )
    print(json.dumps(stats, indent=2))

    if not stats["sufficient_for_training"]:
        _print_section("Insufficient Data for Training")
        print("Valid targets: {0}".format(stats["valid_target_prompts"]))
        print("Minimum required: 5 (3 train + 2 test)")
        print("\nRecommendation: Generate more target data or use synthetic data.")
        return {"status": "insufficient_data", "dataset_stats": stats}

    # --- Step 2: Load and align data for all three predictors ---
    _print_section("Step 2: Data Alignment")
    rows_po, all_ids = load_predictor_targets(features_path, targets_path)
    hs = load_hidden_states(hidden_states_dir)
    rows_hs, _, missing_hs = align_hidden_states_with_targets(hs, targets_path)

    rows_fusion, _, missing_fusion = align_feature_hidden_fusion(
        features_path, targets_path, hidden_states_dir
    )

    print("Prompt-only aligned rows:     {0}".format(len(rows_po)))
    print("Hidden-state-only aligned:    {0}".format(len(rows_hs)))
    print("Fusion aligned rows:          {0}".format(len(rows_fusion)))

    # --- Step 3: Shared train/test split ---
    _print_section("Step 3: Shared Train/Test Split")

    # Use the intersection of all three predictors' available prompts
    # to ensure all are evaluated on the same set.
    po_ids = set(r.prompt_id for r in rows_po)
    hs_ids = set(r.prompt_id for r in rows_hs)
    fusion_ids = set(r.prompt_id for r in rows_fusion)

    common_ids = sorted(po_ids & hs_ids & fusion_ids)
    print("Common prompts across all baselines: {0}".format(len(common_ids)))

    if len(common_ids) < 5:
        # Do NOT fall back to independent splits — that would compare models
        # on different test prompts, making the comparison invalid.
        result: Dict[str, Any] = {
            "status": "INSUFFICIENT_DATA",
            "dataset_stats": stats,
            "split_info": None,
            "metrics": None,
            "model_path": None,
            "warning": (
                "Only {0} prompts are common across all three baselines "
                "(eligible for prompt-only, hidden-state-only, and fusion). "
                "A minimum of {1} common prompts ({2} train + {3} test) is required "
                "for a valid shared-split comparison. The dataset is NOT ready for "
                "training. Missing data details: "
                "prompt-only eligible={4}, hidden-state-only eligible={5}, "
                "fusion eligible={6}. To proceed, run additional generations and "
                "Gemini evaluations to produce valid empirical-risk targets "
                "(empirical_hallucination_risk must be non-null)."
            ).format(
                len(common_ids),
                MIN_TRAIN_PROMPTS + MIN_TEST_PROMPTS,
                MIN_TRAIN_PROMPTS,
                MIN_TEST_PROMPTS,
                len(rows_po),
                len(rows_hs),
                len(rows_fusion),
            ),
            "missing": {
                "prompt_only_eligible": len(rows_po),
                "hidden_state_only_eligible": len(rows_hs),
                "fusion_eligible": len(rows_fusion),
                "common_eligible": len(common_ids),
                "minimum_common_required": MIN_TRAIN_PROMPTS + MIN_TEST_PROMPTS,
            },
        }
        _print_section("INSUFFICIENT_DATA — Shared Split Required")
        print(result["warning"])
        print("\nNo models trained. No metrics reported.")
        print("To test the comparison implementation, use a synthetic fixture:")
        print('  python run_predictor.py compare --features-path <synthetic>')
        return result
    else:
        train_ids, test_ids = train_test_split_by_prompt(
            common_ids, test_size=test_size, random_state=random_state
        )
        print("Train prompts: {0} ({1})".format(len(train_ids), train_ids))
        print("Test prompts:  {0} ({1})".format(len(test_ids), test_ids))

    # --- Step 4: Train and evaluate each predictor ---
    comparison: Dict[str, Any] = {
        "dataset_stats": stats,
        "interpretation_warning": (
            "All eligible empirical-risk targets are identical; fitted metrics "
            "cannot demonstrate predictive discrimination or improvement."
            if stats["distinct_target_values"] < 2
            else None
        ),
        "shared_split": {
            "train_prompt_ids": train_ids,
            "test_prompt_ids": test_ids,
            "train_count": len(train_ids) if train_ids else 0,
            "test_count": len(test_ids) if test_ids else 0,
            "test_size": test_size,
            "random_state": random_state,
        },
        "target_distribution": {
            "common": _target_distribution(rows_po, common_ids),
            "train": _target_distribution(rows_po, train_ids),
            "test": _target_distribution(rows_po, test_ids),
        },
        "predictors": {},
    }

    # --- Prompt-Only ---
    _print_section("Step 4a: Prompt-Only Baseline (Ridge on 5 features)")
    print("Features: {0}".format(ALL_FEATURE_NAMES))
    result_po = _train_prompt_only(
        rows_po, train_ids, test_ids, output_dir,
        test_size, random_state
    )
    comparison["predictors"]["prompt_only"] = result_po

    # --- Hidden-State-Only ---
    _print_section("Step 4b: Hidden-State-Only Baseline (Ridge on PCA(50))")
    print("Representation: {0}".format(HIDDEN_STATE_STRATEGY))
    result_hs = _train_hidden_only(
        rows_hs, train_ids, test_ids, output_dir,
        test_size, random_state
    )
    comparison["predictors"]["hidden_only"] = result_hs

    # --- Fusion ---
    _print_section("Step 4c: Fusion Baseline (Ridge on [5 + 50] features)")
    result_fusion = _train_fusion(
        rows_fusion, train_ids, test_ids, output_dir,
        test_size, random_state
    )
    comparison["predictors"]["fusion"] = result_fusion

    # --- Step 5: Comparison table ---
    _print_section("Step 5: Comparison Summary")
    _print_comparison_table(comparison)

    # Save comparison results
    comparison_path = Path(output_dir) / "comparison_results.json"
    with open(comparison_path, "w") as f:
        json.dump(comparison, f, indent=2)
    print("\nComparison results saved to: {0}".format(comparison_path))

    return comparison


def _train_prompt_only(
    rows: List[TargetRow],
    train_ids: Optional[List[str]],
    test_ids: Optional[List[str]],
    output_dir: Path,
    test_size: float,
    random_state: int,
) -> Dict[str, Any]:
    """Train the prompt-only baseline with optional shared split."""
    from src.predictor.baselines import _extract_feature_vector

    if train_ids is None or test_ids is None:
        result = run_prompt_only_baseline(
            features_path=DEFAULT_FEATURES_PATH,
            targets_path=DEFAULT_TARGETS_PATH,
            output_dir=output_dir / "prompt_only",
            test_size=test_size,
            random_state=random_state,
        )
    else:
        if len(rows) < 5:
            return {"status": "insufficient_data", "model_type": "prompt_only"}

        prompt_ids = [r.prompt_id for r in rows]
        train_idx = [prompt_ids.index(pid) for pid in train_ids if pid in prompt_ids]
        test_idx = [prompt_ids.index(pid) for pid in test_ids if pid in prompt_ids]

        if len(train_idx) < MIN_TRAIN_PROMPTS or len(test_idx) < MIN_TEST_PROMPTS:
            return {"status": "insufficient_data", "model_type": "prompt_only"}

        train_features = np.array([_extract_feature_vector(
            [r for r in rows if r.prompt_id == pid][0].features
        ) for pid in train_ids if pid in prompt_ids])
        train_targets = np.array([
            [r for r in rows if r.prompt_id == pid][0].risk
            for pid in train_ids if pid in prompt_ids
        ])
        test_features = np.array([_extract_feature_vector(
            [r for r in rows if r.prompt_id == pid][0].features
        ) for pid in test_ids if pid in prompt_ids])
        test_targets = np.array([
            [r for r in rows if r.prompt_id == pid][0].risk
            for pid in test_ids if pid in prompt_ids
        ])

        from src.predictor.baselines import SplitResult
        split = SplitResult(
            train_ids=[pid for pid in train_ids if pid in prompt_ids],
            test_ids=[pid for pid in test_ids if pid in prompt_ids],
            train_features=train_features,
            train_targets=train_targets,
            test_features=test_features,
            test_targets=test_targets,
            feature_names=list(ALL_FEATURE_NAMES),
        )

        predictor = PromptOnlyPredictor(alpha=1.0, random_state=random_state)
        predictor.fit(split)
        metrics = predictor.evaluate(split)
        model_path = predictor.save(output_dir / "prompt_only")

        result = {
            "status": "trained",
            "model_type": "prompt_only",
            "metrics": {
                "mae": metrics.mae,
                "rmse": metrics.rmse,
                "r2": metrics.r2,
                "baseline_mae": metrics.baseline_mae,
                "baseline_rmse": metrics.baseline_rmse,
                "baseline_r2": metrics.baseline_r2,
                "n_train": metrics.n_train,
                "n_test": metrics.n_test,
                "improvement_mae_over_baseline": metrics.baseline_mae - metrics.mae,
                "uncertainty": _bootstrap_metric_intervals(
                    metrics.test_targets, metrics.predictions,
                    metrics.baseline_predictions,
                    random_state=random_state,
                ),
            },
            "model_path": str(model_path),
        }

    _print_metrics(result)
    return result


def _train_hidden_only(
    rows,
    train_ids: Optional[List[str]],
    test_ids: Optional[List[str]],
    output_dir: Path,
    test_size: float,
    random_state: int,
) -> Dict[str, Any]:
    """Train the hidden-state-only baseline with optional shared split."""
    if train_ids is None or test_ids is None:
        result = run_hidden_only_baseline(
            targets_path=DEFAULT_TARGETS_PATH,
            hidden_states_dir=DEFAULT_HIDDEN_STATES_DIR,
            output_dir=output_dir / "hidden_only",
            test_size=test_size,
            random_state=random_state,
        )
    else:
        if len(rows) < 5:
            return {"status": "insufficient_data", "model_type": "hidden_only"}

        prompt_ids = [r.prompt_id for r in rows]
        train_prompt_ids = [pid for pid in train_ids if pid in prompt_ids]
        test_prompt_ids = [pid for pid in test_ids if pid in prompt_ids]

        if len(train_prompt_ids) < MIN_TRAIN_PROMPTS or len(test_prompt_ids) < MIN_TEST_PROMPTS:
            return {"status": "insufficient_data", "model_type": "hidden_only"}

        predictor = HiddenStateOnlyPredictor(
            alpha=1.0, pca_components=50, random_state=random_state
        )
        predictor.fit(rows, train_prompt_ids, test_prompt_ids)
        metrics = predictor.evaluate(rows)
        model_path = predictor.save(output_dir / "hidden_only")

        result = {
            "status": "trained",
            "model_type": "hidden_only",
            "metrics": {
                "mae": metrics.mae,
                "rmse": metrics.rmse,
                "r2": metrics.r2,
                "baseline_mae": metrics.baseline_mae,
                "baseline_rmse": metrics.baseline_rmse,
                "baseline_r2": metrics.baseline_r2,
                "n_train": metrics.n_train,
                "n_test": metrics.n_test,
                "improvement_mae_over_baseline": metrics.baseline_mae - metrics.mae,
                "uncertainty": _bootstrap_metric_intervals(
                    metrics.test_targets, metrics.predictions,
                    metrics.baseline_predictions,
                    random_state=random_state,
                ),
            },
            "model_path": str(model_path),
        }

    _print_metrics(result)
    return result


def _train_fusion(
    rows,
    train_ids: Optional[List[str]],
    test_ids: Optional[List[str]],
    output_dir: Path,
    test_size: float,
    random_state: int,
) -> Dict[str, Any]:
    """Train the fusion baseline with optional shared split."""
    if train_ids is None or test_ids is None:
        result = run_fusion_baseline(
            features_path=DEFAULT_FEATURES_PATH,
            targets_path=DEFAULT_TARGETS_PATH,
            hidden_states_dir=DEFAULT_HIDDEN_STATES_DIR,
            output_dir=output_dir / "fusion",
            test_size=test_size,
            random_state=random_state,
        )
    else:
        if len(rows) < 5:
            return {"status": "insufficient_data", "model_type": "fusion"}

        prompt_ids = [r.prompt_id for r in rows]
        train_prompt_ids = [pid for pid in train_ids if pid in prompt_ids]
        test_prompt_ids = [pid for pid in test_ids if pid in prompt_ids]

        if len(train_prompt_ids) < MIN_TRAIN_PROMPTS or len(test_prompt_ids) < MIN_TEST_PROMPTS:
            return {"status": "insufficient_data", "model_type": "fusion"}

        predictor = FusionPredictor(
            alpha=1.0, pca_components=50, random_state=random_state
        )
        predictor.fit(rows, train_prompt_ids, test_prompt_ids)
        metrics = predictor.evaluate(rows)
        model_path = predictor.save(output_dir / "fusion")

        result = {
            "status": "trained",
            "model_type": "fusion",
            "metrics": {
                "mae": metrics.mae,
                "rmse": metrics.rmse,
                "r2": metrics.r2,
                "baseline_mae": metrics.baseline_mae,
                "baseline_rmse": metrics.baseline_rmse,
                "baseline_r2": metrics.baseline_r2,
                "n_train": metrics.n_train,
                "n_test": metrics.n_test,
                "improvement_mae_over_baseline": metrics.baseline_mae - metrics.mae,
                "uncertainty": _bootstrap_metric_intervals(
                    metrics.test_targets, metrics.predictions,
                    metrics.baseline_predictions,
                    random_state=random_state,
                ),
            },
            "model_path": str(model_path),
        }

    _print_metrics(result)
    return result


def _print_metrics(result: Dict[str, Any]) -> None:
    """Print metrics for a single predictor result."""
    if result.get("status") != "trained":
        print("Status: {0} ({1})".format(result.get("status"), result.get("model_type")))
        if "warning" in result:
            print("Warning: {0}".format(result["warning"]))
        return

    m = result["metrics"]
    print("Status: trained")
    print("  Train prompts: {0}".format(m["n_train"]))
    print("  Test prompts:  {0}".format(m["n_test"]))
    print("  MAE:  {0:.4f}  (baseline: {1:.4f})".format(m["mae"], m["baseline_mae"]))
    rmse = m.get("rmse", 0.0)
    base_rmse = m.get("baseline_rmse", 0.0)
    print("  RMSE: {0:.4f}  (baseline: {1:.4f})".format(rmse, base_rmse))
    r2 = m.get("r2")
    r2_base = m.get("baseline_r2")
    r2_str = "{0:.4f}".format(r2) if r2 is not None else "N/A (zero variance)"
    r2_base_str = "{0:.4f}".format(r2_base) if r2_base is not None else "N/A"
    print("  R²:   {0}  (baseline: {1})".format(r2_str, r2_base_str))
    print("  Improvement MAE over baseline: {0:+.4f}".format(
        m.get("improvement_mae_over_baseline", 0.0)
    ))


def _print_comparison_table(comparison: Dict[str, Any]) -> None:
    """Print a comparison table across all predictors."""
    predictors = comparison.get("predictors", {})
    names = ["prompt_only", "hidden_only", "fusion"]
    labels = ["Prompt-Only", "Hidden-State-Only", "Fusion"]

    print("{0:<20} {1:>8} {2:>8} {3:>8} {4:>12} {5:>12}".format(
        "Model", "MAE", "RMSE", "R^2", "dMAE_vs_base", "Trained?"
    ))
    print("-" * 80)

    for name, label in zip(names, labels):
        r = predictors.get(name, {})
        if r.get("status") != "trained":
            print("{0:<20} {'N/A':>8} {'N/A':>8} {'N/A':>8} {'N/A':>12} {0:<12}".format(
                label, "No"
            ))
            continue

        m = r["metrics"]
        r2_val = "{0:.4f}".format(m["r2"]) if m["r2"] is not None else "N/A"
        print("{0:<20} {1:>8.4f} {2:>8.4f} {3:>8} {4:>+12.4f} {5:<12}".format(
            label,
            m["mae"],
            m["rmse"],
            r2_val,
            m["improvement_mae_over_baseline"],
            "Yes",
        ))


def run_report() -> None:
    """Generate a brief report from existing comparison results."""
    comparison_path = DEFAULT_OUTPUT_DIR / "comparison_results.json"
    if not comparison_path.exists():
        print("No comparison results found at {0}".format(comparison_path))
        print("Run 'python run_predictor.py compare' first.")
        return

    with open(comparison_path, "r") as f:
        comparison = json.load(f)

    _print_header("LUX Predictor Comparison Report")
    _print_comparison_table(comparison)

    stats = comparison.get("predictors", {})
    print("\nModels trained on disk:")
    for model_type in ["prompt_only", "hidden_only", "fusion"]:
        model_dir = DEFAULT_OUTPUT_DIR / model_type
        if model_dir.exists():
            print("  {0}: {1}".format(model_type, model_dir))


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="LUX Hallucination-Risk Predictor Comparison CLI"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    compare_parser = subparsers.add_parser("compare", help="Train and compare all baselines")
    compare_parser.add_argument("--features-path", type=Path, default=DEFAULT_FEATURES_PATH)
    compare_parser.add_argument("--targets-path", type=Path, default=DEFAULT_TARGETS_PATH)
    compare_parser.add_argument("--hidden-states-dir", type=Path, default=DEFAULT_HIDDEN_STATES_DIR)
    compare_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    compare_parser.add_argument("--test-size", type=float, default=0.3)
    compare_parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)

    subparsers.add_parser("report", help="Print a report from existing comparison results")

    args = parser.parse_args()

    if args.command == "compare":
        run_comparison(
            features_path=args.features_path,
            targets_path=args.targets_path,
            hidden_states_dir=args.hidden_states_dir,
            output_dir=args.output_dir,
            test_size=args.test_size,
            random_state=args.random_state,
        )
    elif args.command == "report":
        run_report()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
