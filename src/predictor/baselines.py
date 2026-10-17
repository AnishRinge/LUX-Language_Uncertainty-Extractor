"""Prompt-only baseline predictor for LUX hallucination-risk prediction.

This module implements the prompt-only baseline: a regression model that
predicts empirical hallucination risk using ONLY features available before
the target LLM (Qwen3) generates its response. It uses pre-generation prompt
features (length, token counts, question structure) and excludes all
post-generation information (generated answers, labels, hidden states,
evaluator outputs, risk values, answerability flags).

Model choice: Ridge regression (L2-regularized linear regression) wrapped in
a scikit-learn Pipeline with StandardScaler. Chosen over logistic regression
because the target is a continuous risk ratio in [0, 1], and over more
complex models per the "no extensive hyperparameter tuning" constraint.

Train/test split: by prompt ID (not by generation seed). No prompt may
appear in both training and test sets, preventing leakage from multiple
generations of the same prompt.

Usage:
    from src.predictor.baselines import run_prompt_only_baseline
    result = run_prompt_only_baseline(
        features_path="data/production/predictor_features.json",
        targets_path="data/production/targets.json",
        output_dir="data/production/predictor_models",
    )
"""

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Lazy-import sklearn so the module can be imported and unit-tested in
# environments where scikit-learn may not be installed.
try:
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split as sk_train_test_split
    from sklearn.pipeline import Pipeline as SkPipeline
    from sklearn.preprocessing import StandardScaler
    import joblib

    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PROD_DIR = ROOT / "data" / "production"
DEFAULT_FEATURES_PATH = PROD_DIR / "predictor_features.json"
DEFAULT_TARGETS_PATH = PROD_DIR / "targets.json"
DEFAULT_OUTPUT_DIR = PROD_DIR / "predictor_models"
DEFAULT_HIDDEN_STATES_DIR = PROD_DIR / "hidden_states"
DEFAULT_RANDOM_STATE = 42

MIN_TRAIN_PROMPTS = 3
MIN_TEST_PROMPTS = 2
MIN_TOTAL_FOR_SPLIT = MIN_TRAIN_PROMPTS + MIN_TEST_PROMPTS

# --------------------------------------------------------------------------
# Shared exceptions and data classes
# --------------------------------------------------------------------------
class AlignmentError(Exception):
    """Raised when features and targets cannot be aligned by stable ID."""


@dataclass
class HiddenStateRow:
    """A hidden-state representation aligned with a risk target."""
    prompt_id: str
    dataset_index: int
    representation: np.ndarray  # hidden-state feature vector
    risk: float  # empirical_hallucination_risk


@dataclass
class FusionRow:
    """Aligned prompt features + hidden-state representation + target."""
    prompt_id: str
    dataset_index: int
    prompt_features: Dict[str, Any]  # prompt_metrics (excluding forbidden fields)
    hidden_state: np.ndarray
    risk: float


# Feature columns used by the prompt-only baseline (pre-generation only).
# These are derived from prompt_metrics in the predictor features artifact.
NUMERICAL_FEATURES = [
    "question_chars",
    "context_chars",
    "question_tokens",
    "context_tokens",
]

BINARY_FEATURES = [
    "question_ends_with_question_mark",
]

ALL_FEATURE_NAMES = NUMERICAL_FEATURES + BINARY_FEATURES

# Fields that must NEVER appear in the predictor features artifact (post-g
# generation data). Their presence indicates a leakage bug.
LEAKAGE_FORBIDDEN_FIELDS = [
    "generated_answer",
    "generation_label",
    "claim_evaluations",
    "evaluation_status",
    "empirical_hallucination_risk",
    "status",
    "latency",
    "generated_token_count",
    "reached_max_tokens",
]

# Fields excluded from the model's feature vector (may legitimately appear
# in the predictor features artifact as metadata, but are NOT model inputs).
# This includes ground-truth labels, hidden-state references, and provenance.
EXCLUDED_FROM_FEATURES = [
    "answerability",                  # ground-truth SQuAD label (not a prompt feature)
    "hidden_state_ref",               # hidden-state reference (not prompt-only)
    "dataset_source",                 # dataset provenance
    "seed",                           # sampling metadata
    "model_name",                     # target-model identity
    "model_revision",                 # target-model identity
    "prompt_text",                    # full rendered prompt (too verbose)
] + LEAKAGE_FORBIDDEN_FIELDS


@dataclass
class TargetRow:
    """A single aligned feature+target row at the prompt level."""
    prompt_id: str
    dataset_index: int
    features: Dict[str, Any]  # prompt_metrics subset
    risk: float  # empirical_hallucination_risk


@dataclass
class SplitResult:
    """Train/test split result with prompt-level disjointness."""
    train_ids: List[str]
    test_ids: List[str]
    train_features: np.ndarray
    train_targets: np.ndarray
    test_features: np.ndarray
    test_targets: np.ndarray
    feature_names: List[str]

    def assert_disjoint(self) -> None:
        train_set = set(self.train_ids)
        test_set = set(self.test_ids)
        overlap = train_set & test_set
        if overlap:
            raise AssertionError(
                "Train/test prompt ID overlap: {0}".format(sorted(overlap))
            )


@dataclass
class MetricsResult:
    """Metrics for the prompt-only predictor."""
    mae: float
    rmse: float
    r2: Optional[float]  # None if denominator is zero
    baseline_mae: float
    baseline_rmse: float
    baseline_r2: Optional[float]
    n_train: int
    n_test: int
    train_mean: float
    test_targets: List[float]
    predictions: List[float]
    baseline_predictions: List[float]
    train_prompt_ids: List[str]
    test_prompt_ids: List[str]


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _extract_feature_vector(metrics: Dict[str, Any]) -> List[float]:
    """Extract the feature vector from a prompt_metrics dict.

    Uses ONLY prompt-level pre-generation features. Excludes answerability
    (ground-truth label), hidden-state refs, and any post-generation fields.
    """
    vector = []
    for col in NUMERICAL_FEATURES:
        val = metrics.get(col)
        if val is None:
            vector.append(0.0)
        else:
            vector.append(float(val))
    for col in BINARY_FEATURES:
        val = metrics.get(col)
        # Convert True/False to 1.0/0.0
        vector.append(1.0 if bool(val) else 0.0)
    return vector


def load_predictor_targets(
    features_path: Path = DEFAULT_FEATURES_PATH,
    targets_path: Path = DEFAULT_TARGETS_PATH,
) -> Tuple[List[TargetRow], List[str]]:
    """Load and align predictor features with empirical-risk targets.

    Returns (rows, all_prompt_ids) where rows contains only prompts with
    valid (non-None) risk targets. all_prompt_ids lists every prompt in
    the feature file for auditing.
    """
    features_data = _load_json(features_path)
    targets_data = _load_json(targets_path)

    feats = features_data.get("predictor_features", [])
    targets_by_idx = {}
    for t in targets_data.get("targets", []):
        if t["dataset_index"] not in targets_by_idx:
            targets_by_idx[t["dataset_index"]] = t  # keep first occurrence

    rows: List[TargetRow] = []
    all_prompt_ids: List[str] = []
    for entry in feats:
        prompt_id = entry["prompt_id"]
        dataset_index = entry["dataset_index"]
        all_prompt_ids.append(prompt_id)

        tgt = targets_by_idx.get(dataset_index)
        if tgt is None:
            continue

        risk = tgt.get("empirical_hallucination_risk")
        if risk is None:
            continue

        # Verify no leakage: post-generation fields must not appear in the
        # feature entry or in prompt_metrics.
        for forbidden in LEAKAGE_FORBIDDEN_FIELDS:
            if forbidden in entry:
                raise ValueError(
                    "Leakage prevention: predictor feature for prompt {0} "
                    "contains forbidden post-generation field '{1}'".format(prompt_id, forbidden)
                )
            if forbidden in entry.get("prompt_metrics", {}):
                raise ValueError(
                    "Leakage prevention: prompt_metrics for prompt {0} "
                    "contains forbidden post-generation field '{1}'".format(prompt_id, forbidden)
                )

        rows.append(TargetRow(
            prompt_id=prompt_id,
            dataset_index=dataset_index,
            features=dict(entry.get("prompt_metrics", {})),
            risk=float(risk),
        ))

    return rows, all_prompt_ids


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    baseline_pred: Optional[np.ndarray] = None,
    train_mean: Optional[float] = None,
) -> Dict[str, Any]:
    """Compute regression metrics: MAE, RMSE, R^2.

    R^2 is None when the denominator (total sum of squares) is zero —
    i.e., all test targets are identical. Returns baseline metrics from
    the training-set mean if baseline_pred is provided.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

    # R^2 is only meaningful when y_true has variance.
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else None

    result = {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "n": int(len(y_true)),
    }

    if baseline_pred is not None and train_mean is not None:
        baseline_pred = np.asarray(baseline_pred, dtype=float)
        result["baseline_mae"] = float(mean_absolute_error(y_true, baseline_pred))
        result["baseline_rmse"] = float(np.sqrt(mean_squared_error(y_true, baseline_pred)))
        ss_res_base = float(np.sum((y_true - baseline_pred) ** 2))
        r2_base = 1.0 - (ss_res_base / ss_tot) if ss_tot > 0 else None
        result["baseline_r2"] = r2_base
        result["baseline_pred_value"] = float(train_mean)
        result["improvement_mae_over_baseline"] = result["baseline_mae"] - mae

    return result


def prompt_only_train_test_split(
    rows: List[TargetRow],
    test_size: float = 0.3,
    random_state: int = DEFAULT_RANDOM_STATE,
    min_train: int = 3,
    min_test: int = 2,
) -> Optional[SplitResult]:
    """Split by prompt ID into train/test sets.

    No prompt appears in both sets. Returns None if there are too few
    valid targets for a meaningful split (fewer than min_train + min_test).

    Args:
        rows: List of TargetRow with aligned features and targets.
        test_size: Fraction of prompts for testing.
        random_state: Reproducibility seed.
        min_train: Minimum training prompts.
        min_test: Minimum test prompts.

    Returns:
        SplitResult or None if insufficient data.
    """
    n = len(rows)
    if n < min_train + min_test:
        return None

    prompt_ids = [r.prompt_id for r in rows]
    features_array = np.array([_extract_feature_vector(r.features) for r in rows])
    targets_array = np.array([r.risk for r in rows])

    train_idx, test_idx = sk_train_test_split(
        list(range(n)),
        test_size=test_size,
        random_state=random_state,
    )

    train_ids = [prompt_ids[i] for i in train_idx]
    test_ids = [prompt_ids[i] for i in test_idx]
    train_features = features_array[train_idx]
    test_features = features_array[test_idx]
    train_targets = targets_array[train_idx]
    test_targets = targets_array[test_idx]

    result = SplitResult(
        train_ids=train_ids,
        test_ids=test_ids,
        train_features=train_features,
        train_targets=train_targets,
        test_features=test_features,
        test_targets=test_targets,
        feature_names=list(ALL_FEATURE_NAMES),
    )
    result.assert_disjoint()
    return result


class PromptOnlyPredictor:
    """Prompt-only hallucination-risk predictor (Ridge regression).

    Uses ONLY pre-generation prompt features — no generated answers, hidden
    states, evaluator outputs, or target values as input.
    """

    def __init__(self, alpha: float = 1.0, random_state: int = DEFAULT_RANDOM_STATE):
        if not _HAS_SKLEARN:
            raise ImportError(
                "scikit-learn is required for PromptOnlyPredictor. "
                "Install with: pip install scikit-learn"
            )
        self.alpha = alpha
        self.random_state = random_state
        self._pipeline: Optional[SkPipeline] = None
        self._feature_names: List[str] = list(ALL_FEATURE_NAMES)
        self._train_prompt_ids: List[str] = []
        self._test_prompt_ids: List[str] = []
        self._train_mean: Optional[float] = None

    def _build_pipeline(self) -> SkPipeline:
        return SkPipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=self.alpha, random_state=self.random_state)),
        ])

    def fit(self, split: SplitResult) -> "PromptOnlyPredictor":
        """Fit the model on training data only.

        Imputers, scalers, and feature selection are fitted on training data
        only to prevent leakage.
        """
        self._train_prompt_ids = list(split.train_ids)
        self._test_prompt_ids = list(split.test_ids)
        self._train_mean = float(np.mean(split.train_targets))
        self._pipeline = self._build_pipeline()
        self._pipeline.fit(split.train_features, split.train_targets)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict risk for a feature matrix."""
        if self._pipeline is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self._pipeline.predict(features)

    def evaluate(self, split: SplitResult) -> MetricsResult:
        """Evaluate on the held-out test set.

        Reports both the Ridge model's metrics and the training-set mean
        baseline on the same test prompts.
        """
        preds = self.predict(split.test_features)

        # Baseline: constant prediction equal to training-set mean risk.
        train_mean = self._train_mean or float(np.mean(split.train_targets))
        baseline_preds = np.full(len(split.test_targets), train_mean)

        m = compute_metrics(
            y_true=split.test_targets,
            y_pred=preds,
            baseline_pred=baseline_preds,
            train_mean=train_mean,
        )

        return MetricsResult(
            mae=m["mae"],
            rmse=m["rmse"],
            r2=m["r2"],
            baseline_mae=m["baseline_mae"],
            baseline_rmse=m["baseline_rmse"],
            baseline_r2=m["baseline_r2"],
            n_train=len(split.train_ids),
            n_test=len(split.test_ids),
            train_mean=train_mean,
            test_targets=[float(x) for x in split.test_targets],
            predictions=[float(x) for x in preds],
            baseline_predictions=[float(x) for x in baseline_preds],
            train_prompt_ids=list(split.train_ids),
            test_prompt_ids=list(split.test_ids),
        )

    def save(self, output_dir: Path) -> Path:
        """Persist the fitted model and metadata."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save the sklearn pipeline via joblib.
        model_path = output_dir / "prompt_only_model.joblib"
        if self._pipeline is not None:
            joblib.dump(self._pipeline, model_path)

        metadata = {
            "model_type": "PromptOnlyPredictor",
            "feature_names": self._feature_names,
            "all_features": ALL_FEATURE_NAMES,
            "excluded_features": EXCLUDED_FROM_FEATURES,
            "model_class": "sklearn.pipeline.Pipeline",
            "steps": [
                {"name": "StandardScaler", "purpose": "Standardize numerical features"},
                {"name": "Ridge", "alpha": self.alpha, "purpose": "L2-regularized regression for continuous risk"},
            ],
            "alpha": self.alpha,
            "random_state": self.random_state,
            "train_prompt_ids": self._train_prompt_ids,
            "test_prompt_ids": self._test_prompt_ids,
            "train_mean": self._train_mean,
            "training_only_preprocessing": True,
            "features_excluded_for_leakage": True,
            "target_variable": "empirical_hallucination_risk",
            "risk_formula": "UNRELIABLE generations / total SUCCESS evaluations",
            "prediction_timing": "pre-generation",
            "sklearn_available": _HAS_SKLEARN,
        }
        meta_path = output_dir / "metadata.json"
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(metadata, fh, indent=2)

        return output_dir / "prompt_only_model.joblib"

    @classmethod
    def load(cls, model_path: Path, metadata_path: Optional[Path] = None) -> "PromptOnlyPredictor":
        """Load a persisted model and metadata."""
        if not _HAS_SKLEARN:
            raise ImportError(
                "scikit-learn is required for PromptOnlyPredictor. "
                "Install with: pip install scikit-learn"
            )
        model_path = Path(model_path)
        pipeline = joblib.load(model_path)

        predictor = cls()
        predictor._pipeline = pipeline
        predictor._feature_names = list(ALL_FEATURE_NAMES)

        if metadata_path is not None:
            metadata_path = Path(metadata_path)
            if metadata_path.exists():
                with open(metadata_path, "r", encoding="utf-8") as fh:
                    meta = json.load(fh)
                predictor._train_prompt_ids = meta.get("train_prompt_ids", [])
                predictor._test_prompt_ids = meta.get("test_prompt_ids", [])
                predictor._train_mean = meta.get("train_mean")

        return predictor

    def get_feature_names(self) -> List[str]:
        """Return the ordered list of feature names used by this predictor."""
        return list(self._feature_names)


def run_prompt_only_baseline(
    features_path: Path = DEFAULT_FEATURES_PATH,
    targets_path: Path = DEFAULT_TARGETS_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    test_size: float = 0.3,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> Dict[str, Any]:
    """End-to-end prompt-only baseline training and evaluation.

    Returns a dict with:
        - dataset_stats: counts of prompts with/without valid targets
        - split_info: train/test prompt IDs (or None if insufficient)
        - metrics: MetricsResult (or None if insufficient data)
        - model_path: path to saved model (or None)
        - status: "trained", "insufficient_data", or "sklearn_unavailable"
    """
    if not _HAS_SKLEARN:
        return {
            "status": "sklearn_unavailable",
            "dataset_stats": _compute_dataset_stats(features_path, targets_path),
        }

    rows, all_prompt_ids = load_predictor_targets(features_path, targets_path)
    stats = _compute_dataset_stats(features_path, targets_path)

    result: Dict[str, Any] = {
        "dataset_stats": stats,
        "split_info": None,
        "metrics": None,
        "model_path": None,
    }

    if len(rows) < 5:
        result["status"] = "insufficient_data"
        result["warning"] = (
            "Only {0} prompts with valid risk targets; minimum of 5 required "
            "for a train/test split (3 train + 2 test). Metrics are not "
            "reported.".format(len(rows))
        )
        return result

    split = prompt_only_train_test_split(
        rows, test_size=test_size, random_state=random_state
    )
    if split is None:
        result["status"] = "insufficient_data"
        result["warning"] = (
            "Only {0} prompts with valid targets; too few for a meaningful "
            "split.".format(len(rows))
        )
        return result

    predictor = PromptOnlyPredictor(alpha=1.0, random_state=random_state)
    predictor.fit(split)
    metrics = predictor.evaluate(split)

    model_path = predictor.save(output_dir)

    predictions = {
        "test_prompt_ids": metrics.test_prompt_ids,
        "test_targets": metrics.test_targets,
        "predictions": metrics.predictions,
        "baseline_predictions": metrics.baseline_predictions,
        "train_mean": metrics.train_mean,
    }
    pred_path = Path(output_dir) / "predictions.json"
    with open(pred_path, "w", encoding="utf-8") as fh:
        json.dump(predictions, fh, indent=2)

    result["status"] = "trained"
    result["split_info"] = {
        "train_prompt_ids": metrics.train_prompt_ids,
        "test_prompt_ids": metrics.test_prompt_ids,
        "train_prompt_count": len(metrics.train_prompt_ids),
        "test_prompt_count": len(metrics.test_prompt_ids),
        "feature_names": predictor.get_feature_names(),
    }
    result["metrics"] = {
        "mae": metrics.mae,
        "rmse": metrics.rmse,
        "r2": metrics.r2,
        "baseline_mae": metrics.baseline_mae,
        "baseline_rmse": metrics.baseline_rmse,
        "baseline_r2": metrics.baseline_r2,
        "improvement_mae_over_baseline": metrics.baseline_mae - metrics.mae,
        "train_mean": metrics.train_mean,
        "n_train": metrics.n_train,
        "n_test": metrics.n_test,
    }
    result["model_path"] = str(model_path)
    result["predictions_path"] = str(pred_path)
    return result


def _compute_dataset_stats(
    features_path: Path = DEFAULT_FEATURES_PATH,
    targets_path: Path = DEFAULT_TARGETS_PATH,
) -> Dict[str, Any]:
    """Recalculate dataset statistics from the actual on-disk files."""
    rows, all_prompt_ids = load_predictor_targets(features_path, targets_path)
    total_prompts = len(all_prompt_ids)
    valid_prompts = len(rows)
    missing = total_prompts - valid_prompts

    risk_values = [r.risk for r in rows]
    distinct = len(set(risk_values))
    risk_range = [min(risk_values), max(risk_values)] if risk_values else None

    sufficient = valid_prompts >= 5

    return {
        "total_prompts": total_prompts,
        "valid_target_prompts": valid_prompts,
        "missing_target_prompts": missing,
        "distinct_target_values": distinct,
        "risk_range": risk_range,
        "risk_values": sorted(risk_values),
        "sufficient_for_training": sufficient,
    }


# --------------------------------------------------------------------------
# Generic train/test split by prompt ID (shared by all baselines)
# --------------------------------------------------------------------------
def train_test_split_by_prompt(
    prompt_ids: List[str],
    test_size: float = 0.3,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> Tuple[List[str], List[str]]:
    """Split prompt IDs into train/test sets with no overlap.

    Returns (train_ids, test_ids). Each prompt appears in exactly one set.
    """
    n = len(prompt_ids)
    if n < MIN_TOTAL_FOR_SPLIT:
        raise AlignmentError(
            "Insufficient prompts ({0}) for train/test split; "
            "minimum {1} required ({2} train + {3} test).".format(
                n, MIN_TOTAL_FOR_SPLIT, MIN_TRAIN_PROMPTS, MIN_TEST_PROMPTS
            )
        )
    train_idx, test_idx = sk_train_test_split(
        list(range(n)),
        test_size=test_size,
        random_state=random_state,
    )
    train_ids = [prompt_ids[i] for i in train_idx]
    test_ids = [prompt_ids[i] for i in test_idx]
    # Programmatically verify disjointness.
    overlap = set(train_ids) & set(test_ids)
    if overlap:
        raise AlignmentError(
            "Train/test overlap (should never happen): {0}".format(sorted(overlap))
        )
    return train_ids, test_ids


# --------------------------------------------------------------------------
# Hidden-state loading and representation extraction
# --------------------------------------------------------------------------
def load_hidden_states(hidden_states_dir: Path = DEFAULT_HIDDEN_STATES_DIR) -> Dict[int, Dict[str, Any]]:
    """Load all hidden-state artifacts from disk.

    Returns a dict keyed by ``prompt_dataset_index`` (stable prompt ID)
    mapping to the raw hidden-state payload. Does NOT regenerate or recompute
    any representations — only reads existing files.

    Raises FileNotFoundError if the directory does not exist.
    """
    hidden_states_dir = Path(hidden_states_dir)
    if not hidden_states_dir.exists():
        return {}
    result: Dict[int, Dict[str, Any]] = {}
    for path in sorted(hidden_states_dir.glob("hs_*.json")):
        try:
            data = _load_json(path)
            ds_idx = data["prompt_dataset_index"]
            result[ds_idx] = data
        except (json.JSONDecodeError, KeyError, OSError):
            continue
    return result


def extract_hidden_representation(
    hidden_state_data: Dict[str, Any],
    strategy: str = "last_layer_mean_pooled",
) -> np.ndarray:
    """Extract a fixed-dimensional representation from a hidden-state payload.

    Strategy:
      - ``last_layer_mean_pooled`` (default): Use the mean-pooled tokens
        representation from the last transformer layer (index -1, excluding
        the embedding layer at index 0). This is a single vector of
        ``hidden_size`` dimensions (2048 for Qwen3-1.7B).

    The hidden-state files store ``mean_pooled`` as a list of ``num_layers``
    vectors, each of length ``hidden_size``. We select the last layer's
    vector.
    """
    if strategy != "last_layer_mean_pooled":
        raise ValueError("Unknown hidden-state strategy: {0}".format(strategy))

    mean_pooled = hidden_state_data.get("mean_pooled")
    if mean_pooled is None:
        raise AlignmentError(
            "Hidden-state payload missing 'mean_pooled' field"
        )

    layers = np.array(mean_pooled, dtype=np.float32)
    if layers.ndim != 2:
        raise AlignmentError(
            "Expected 2D mean_pooled array, got shape {0}".format(layers.shape)
        )

    # Return the last layer's representation.
    return layers[-1]


def _compute_dataset_stats_with_hidden(
    features_path: Path = DEFAULT_FEATURES_PATH,
    targets_path: Path = DEFAULT_TARGETS_PATH,
    hidden_states_dir: Path = DEFAULT_HIDDEN_STATES_DIR,
) -> Dict[str, Any]:
    """Extended dataset stats including hidden-state availability."""
    base = _compute_dataset_stats(features_path, targets_path)
    hs = load_hidden_states(hidden_states_dir)
    # Compute feature dataset indices for intersection.
    feat_data = _load_json(features_path)
    feat_ds_indices = set()
    for entry in feat_data.get("predictor_features", []):
        feat_ds_indices.add(entry.get("dataset_index"))
    base["hidden_state_files"] = len(hs)
    base["prompts_with_hidden_states"] = len(hs)
    base["prompts_with_both_features_and_hidden"] = len(
        feat_ds_indices & set(hs.keys())
    )
    return base