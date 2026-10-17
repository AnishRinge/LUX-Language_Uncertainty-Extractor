"""Hidden-state-only baseline predictor for LUX hallucination-risk prediction.

Uses ONLY pre-generation Qwen3 hidden-state representations to predict
empirical hallucination risk. No prompt-text features, generated answers,
or evaluator outputs are used as input.

Hidden-state representation:
  - Loaded from on-disk JSON files in ``data/production/hidden_states/``
  - Strategy: last transformer layer's mean-pooled token representation
    (2048 dimensions for Qwen3-1.7B, 29 layers total including embeddings)
  - Dimensionality reduction: PCA to 50 components (fitted on training data only)

Model: Ridge regression (same family as the prompt-only baseline) within a
scikit-learn Pipeline with StandardScaler + PCA + Ridge.

Usage:
    from src.predictor.hidden_only import run_hidden_only_baseline
    result = run_hidden_only_baseline()
"""

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.predictor.baselines import (  # noqa: E402
    DEFAULT_HIDDEN_STATES_DIR,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_RANDOM_STATE,
    DEFAULT_TARGETS_PATH,
    AlignmentError,
    HiddenStateRow,
    MetricsResult,
    compute_metrics,
    extract_hidden_representation,
    load_hidden_states,
    train_test_split_by_prompt,
)

try:
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline as SkPipeline
    from sklearn.preprocessing import StandardScaler
    import joblib

    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False

# Hidden-state representation configuration.
HIDDEN_STATE_STRATEGY = "last_layer_mean_pooled"
PCA_N_COMPONENTS = 50
RIDGE_ALPHA = 1.0


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def align_hidden_states_with_targets(
    hidden_states: Dict[int, Dict[str, Any]],
    targets_path: Path = DEFAULT_TARGETS_PATH,
) -> Tuple[List[HiddenStateRow], List[int], List[int]]:
    """Align hidden-state files with valid empirical-risk targets.

    Returns (aligned_rows, all_ds_indices, missing_ds_indices) where
    aligned_rows contains only prompts that have BOTH a hidden-state file
    AND a valid (non-None) risk target.
    """
    targets_data = _load_json(targets_path)
    targets_by_idx = {}
    for t in targets_data.get("targets", []):
        if t["dataset_index"] not in targets_by_idx:
            targets_by_idx[t["dataset_index"]] = t

    all_ds_indices = sorted(hidden_states.keys())
    aligned: List[HiddenStateRow] = []
    missing: List[int] = []

    for ds_idx in all_ds_indices:
        tgt = targets_by_idx.get(ds_idx)
        if tgt is None:
            missing.append(ds_idx)
            continue
        risk = tgt.get("empirical_hallucination_risk")
        if risk is None:
            missing.append(ds_idx)
            continue
        prompt_id = tgt.get("prompt_id", "prompt_{0}".format(ds_idx))
        rep = extract_hidden_representation(
            hidden_states[ds_idx],
            strategy=HIDDEN_STATE_STRATEGY,
        )
        aligned.append(HiddenStateRow(
            prompt_id=prompt_id,
            dataset_index=ds_idx,
            representation=rep,
            risk=float(risk),
        ))

    return aligned, all_ds_indices, missing


def build_hidden_state_matrix(rows: List[HiddenStateRow]) -> np.ndarray:
    """Build a 2D feature matrix from hidden-state representations.

    Each row is one prompt's hidden-state feature vector. All vectors must
    have the same dimensionality; mismatched dimensions raise AlignmentError.
    """
    dims = set(len(r.representation) for r in rows)
    if len(dims) > 1:
        raise AlignmentError(
            "Hidden-state dimension mismatch: found {0} different dims".format(
                sorted(dims)
            )
        )
    return np.array([r.representation for r in rows], dtype=np.float32)


class HiddenStateOnlyPredictor:
    """Hidden-state-only hallucination-risk predictor.

    Uses ONLY pre-generation hidden-state representations — no prompt-text
    features, generated answers, or evaluator outputs.
    """

    def __init__(
        self,
        alpha: float = RIDGE_ALPHA,
        pca_components: int = PCA_N_COMPONENTS,
        random_state: int = DEFAULT_RANDOM_STATE,
    ):
        if not _HAS_SKLEARN:
            raise ImportError(
                "scikit-learn is required. Install with: pip install scikit-learn"
            )
        self.alpha = alpha
        self.pca_components = pca_components
        self.random_state = random_state
        self._pipeline: Optional[SkPipeline] = None
        self._train_prompt_ids: List[str] = []
        self._test_prompt_ids: List[str] = []
        self._train_mean: Optional[float] = None
        self._hidden_dim: Optional[int] = None

    def _build_pipeline(self, input_dim: int, n_train: int) -> SkPipeline:
        n_components = min(self.pca_components, input_dim - 1, n_train - 1)
        if n_components < 1:
            n_components = 1
        return SkPipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=n_components, random_state=self.random_state)),
            ("ridge", Ridge(alpha=self.alpha, random_state=self.random_state)),
        ])

    def fit(self, rows: List[HiddenStateRow], train_ids: List[str], test_ids: List[str]) -> "HiddenStateOnlyPredictor":
        """Fit the model on training data only.

        PCA and StandardScaler are fitted on training data only to prevent
        leakage. Train/test split is provided externally and shared across
        all baselines.
        """
        train_rows = [r for r in rows if r.prompt_id in set(train_ids)]
        test_rows = [r for r in rows if r.prompt_id in set(test_ids)]

        train_X = build_hidden_state_matrix(train_rows)
        train_y = np.array([r.risk for r in train_rows])
        self._train_mean = float(np.mean(train_y))

        self._hidden_dim = train_X.shape[1]
        self._pipeline = self._build_pipeline(self._hidden_dim, train_X.shape[0])
        self._pipeline.fit(train_X, train_y)

        self._train_prompt_ids = list(train_ids)
        self._test_prompt_ids = list(test_ids)
        return self

    def predict(self, rows: List[HiddenStateRow]) -> np.ndarray:
        """Predict risk for hidden-state rows."""
        if self._pipeline is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        X = build_hidden_state_matrix(rows)
        return self._pipeline.predict(X)

    def evaluate(self, rows: List[HiddenStateRow]) -> MetricsResult:
        """Evaluate on the held-out test set."""
        test_rows = [r for r in rows if r.prompt_id in set(self._test_prompt_ids)]
        test_X = build_hidden_state_matrix(test_rows)
        test_y = np.array([r.risk for r in test_rows])

        preds = self._pipeline.predict(test_X)

        train_mean = self._train_mean
        baseline_preds = np.full(len(test_y), train_mean)

        m = compute_metrics(
            y_true=test_y,
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
            n_train=len(self._train_prompt_ids),
            n_test=len(self._test_prompt_ids),
            train_mean=train_mean,
            test_targets=[float(x) for x in test_y],
            predictions=[float(x) for x in preds],
            baseline_predictions=[float(x) for x in baseline_preds],
            train_prompt_ids=list(self._train_prompt_ids),
            test_prompt_ids=list(self._test_prompt_ids),
        )

    def save(self, output_dir: Path) -> Path:
        """Persist the fitted model and metadata."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        model_path = output_dir / "hidden_only_model.joblib"
        if self._pipeline is not None:
            joblib.dump(self._pipeline, model_path)

        metadata = {
            "model_type": "HiddenStateOnlyPredictor",
            "representation_strategy": HIDDEN_STATE_STRATEGY,
            "hidden_dimension": self._hidden_dim,
            "pca_components": self.pca_components,
            "ridge_alpha": self.alpha,
            "random_state": self.random_state,
            "train_prompt_ids": self._train_prompt_ids,
            "test_prompt_ids": self._test_prompt_ids,
            "train_mean": self._train_mean,
            "training_only_preprocessing": True,
            "uses_prompt_features": False,
            "uses_hidden_states": True,
            "hidden_state_representation": "last_layer_mean_pooled (2048-dim -> PCA -> Ridge)",
            "pca_fitted_on_training_only": True,
            "target_variable": "empirical_hallucination_risk",
            "risk_formula": "UNRELIABLE generations / total SUCCESS evaluations",
            "prediction_timing": "pre-generation",
        }
        meta_path = output_dir / "metadata.json"
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(metadata, fh, indent=2)

        return model_path

    @classmethod
    def load(cls, model_path: Path, metadata_path: Optional[Path] = None) -> "HiddenStateOnlyPredictor":
        """Load a persisted model and metadata."""
        model_path = Path(model_path)
        pipeline = joblib.load(model_path)

        predictor = cls()
        predictor._pipeline = pipeline

        if metadata_path is not None:
            metadata_path = Path(metadata_path)
            if metadata_path.exists():
                with open(metadata_path, "r", encoding="utf-8") as fh:
                    meta = json.load(fh)
                predictor._train_prompt_ids = meta.get("train_prompt_ids", [])
                predictor._test_prompt_ids = meta.get("test_prompt_ids", [])
                predictor._train_mean = meta.get("train_mean")
                predictor._hidden_dim = meta.get("hidden_dimension")

        return predictor

    def get_feature_names(self) -> List[str]:
        """Return descriptive names for the hidden-state feature representation."""
        return ["hidden_state_pca_components"]


def run_hidden_only_baseline(
    targets_path: Path = DEFAULT_TARGETS_PATH,
    hidden_states_dir: Path = DEFAULT_HIDDEN_STATES_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    test_size: float = 0.3,
    random_state: int = DEFAULT_RANDOM_STATE,
    train_ids: Optional[List[str]] = None,
    test_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """End-to-end hidden-state-only baseline training and evaluation.

    If train_ids/test_ids are provided (from a shared split), they are used
    to ensure all baselines are evaluated on the same prompt partition.
    """
    if not _HAS_SKLEARN:
        return {
            "status": "sklearn_unavailable",
            "model_type": "hidden_only",
        }

    hs = load_hidden_states(hidden_states_dir)
    rows, all_ds_indices, missing = align_hidden_states_with_targets(hs, targets_path)

    stats = {
        "model_type": "hidden_only",
        "hidden_state_files_on_disk": len(hs),
        "prompts_with_hidden_states": len(hs),
        "prompts_with_valid_targets": len(rows),
        "prompts_missing_targets": len(missing),
        "sufficient_for_training": len(rows) >= 5,
        "representation": HIDDEN_STATE_STRATEGY,
    }

    result: Dict[str, Any] = {
        "dataset_stats": stats,
        "split_info": None,
        "metrics": None,
        "model_path": None,
        "status": None,
    }

    if len(rows) < 5:
        result["status"] = "insufficient_data"
        result["warning"] = (
            "Only {0} prompts with both hidden states and valid targets; "
            "minimum of 5 required for a train/test split. Metrics are "
            "not reported.".format(len(rows))
        )
        return result

    prompt_ids = [r.prompt_id for r in rows]

    if train_ids is None or test_ids is None:
        train_ids, test_ids = train_test_split_by_prompt(
            prompt_ids, test_size=test_size, random_state=random_state
        )

    predictor = HiddenStateOnlyPredictor(
        alpha=RIDGE_ALPHA,
        pca_components=PCA_N_COMPONENTS,
        random_state=random_state,
    )
    predictor.fit(rows, train_ids, test_ids)
    metrics = predictor.evaluate(rows)

    model_path = predictor.save(output_dir)

    predictions = {
        "model_type": "hidden_state_only",
        "test_prompt_ids": metrics.test_prompt_ids,
        "test_targets": metrics.test_targets,
        "predictions": metrics.predictions,
        "baseline_predictions": metrics.baseline_predictions,
        "train_mean": metrics.train_mean,
        "representation": HIDDEN_STATE_STRATEGY,
    }
    pred_path = Path(output_dir) / "hidden_only_predictions.json"
    with open(pred_path, "w", encoding="utf-8") as fh:
        json.dump(predictions, fh, indent=2)

    result["status"] = "trained"
    result["split_info"] = {
        "train_prompt_ids": metrics.train_prompt_ids,
        "test_prompt_ids": metrics.test_prompt_ids,
        "train_prompt_count": len(metrics.train_prompt_ids),
        "test_prompt_count": len(metrics.test_prompt_ids),
        "feature_description": "last_layer_mean_pooled (PCA-reduced)",
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
