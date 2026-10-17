"""Fusion baseline predictor for LUX hallucination-risk prediction.

Combines pre-generation prompt features and hidden-state representations
into a single Ridge regression model.

Fusion architecture:
  - Prompt features (5 numerical features from prompt_metrics): question_chars,
    context_chars, question_ends_with_question_mark, question_tokens, context_tokens
    (answerability is excluded — it is a ground-truth label, not a pre-generation
    feature)
  - Hidden-state features: last-layer mean-pooled representation (2048-dim)
    reduced to 50 PCA components (fitted on training data only)
  - Combined: 5-dim prompt features + 50-dim PCA hidden states = 55-dim vector
  - Preprocessing: StandardScaler on the combined vector (fitted on train only)
  - Model: Ridge regression

Usage:
    from src.predictor.fusion import run_fusion_baseline
    result = run_fusion_baseline()
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
    DEFAULT_FEATURES_PATH,
    DEFAULT_HIDDEN_STATES_DIR,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_RANDOM_STATE,
    DEFAULT_TARGETS_PATH,
    AlignmentError,
    FusionRow,
    MetricsResult,
    compute_metrics,
    extract_hidden_representation,
    load_hidden_states,
    train_test_split_by_prompt,
)

# Reuse prompt-only feature definitions.
from src.predictor.baselines import (  # noqa: E402
    ALL_FEATURE_NAMES as FEATURE_COLUMNS,
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

HIDDEN_STATE_STRATEGY = "last_layer_mean_pooled"
PCA_N_COMPONENTS = 50
RIDGE_ALPHA = 1.0


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def align_feature_hidden_fusion(
    features_path: Path = DEFAULT_FEATURES_PATH,
    targets_path: Path = DEFAULT_TARGETS_PATH,
    hidden_states_dir: Path = DEFAULT_HIDDEN_STATES_DIR,
) -> Tuple[List[FusionRow], List[int], List[int]]:
    """Align prompt features, hidden states, and risk targets.

    Returns (aligned_rows, all_ds_indices, missing_ds_indices) where
    aligned_rows contains only prompts that have ALL of:
      - a prompt feature entry (with valid prompt_metrics)
      - a hidden-state file on disk
      - a valid (non-None) empirical risk target
    """
    features_data = _load_json(features_path)
    targets_data = _load_json(targets_path)
    hidden_states = load_hidden_states(hidden_states_dir)

    targets_by_idx = {}
    for t in targets_data.get("targets", []):
        if t["dataset_index"] not in targets_by_idx:
            targets_by_idx[t["dataset_index"]] = t

    features_by_idx = {}
    features_list = (
        features_data.get("predictor_features", [])
        if "predictor_features" in features_data
        else features_data.get("features", [])
        if "features" in features_data
        else features_data.get("predictions", [])
    )
    for f in features_list:
        ds_idx = f.get("dataset_index")
        if ds_idx is not None and ds_idx not in features_by_idx:
            features_by_idx[ds_idx] = f

    all_ds_indices = sorted(set(features_by_idx.keys()) | set(hidden_states.keys()))
    aligned: List[FusionRow] = []
    missing: List[int] = []

    for ds_idx in all_ds_indices:
        feat = features_by_idx.get(ds_idx)
        tgt = targets_by_idx.get(ds_idx)
        hs_data = hidden_states.get(ds_idx)

        if feat is None or tgt is None or hs_data is None:
            missing.append(ds_idx)
            continue

        risk = tgt.get("empirical_hallucination_risk")
        if risk is None:
            missing.append(ds_idx)
            continue

        prompt_metrics = feat.get("prompt_metrics", {})
        hidden_rep = extract_hidden_representation(hs_data, strategy=HIDDEN_STATE_STRATEGY)

        prompt_id = tgt.get("prompt_id", feat.get("prompt_id", "prompt_{0}".format(ds_idx)))

        aligned.append(FusionRow(
            prompt_id=prompt_id,
            dataset_index=ds_idx,
            prompt_features=prompt_metrics,
            hidden_state=hidden_rep,
            risk=float(risk),
        ))

    return aligned, all_ds_indices, missing


def build_fusion_matrix(rows: List[FusionRow]) -> Tuple[np.ndarray, np.ndarray]:
    """Build feature matrices for prompt features and hidden states.

    Returns (prompt_features_matrix, hidden_state_matrix).
    - prompt_features_matrix: (n_samples, len(FEATURE_COLUMNS)) float32
    - hidden_state_matrix: (n_samples, hidden_dim) float32
    """
    prompt_data = []
    hidden_data = []
    for r in rows:
        row_vec = []
        for col in FEATURE_COLUMNS:
            val = r.prompt_features.get(col, 0.0)
            row_vec.append(float(val))
        prompt_data.append(row_vec)
        hidden_data.append(np.asarray(r.hidden_state, dtype=np.float32))

    prompt_matrix = np.array(prompt_data, dtype=np.float32)
    hidden_matrix = np.array(hidden_data, dtype=np.float32)

    if hidden_matrix.ndim == 1:
        hidden_matrix = hidden_matrix.reshape(1, -1)

    return prompt_matrix, hidden_matrix


class FusionPredictor:
    """Fusion hallucination-risk predictor combining prompt + hidden-state features.

    Uses BOTH pre-generation prompt features AND hidden-state representations.
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
        self._scaler_X: Optional[StandardScaler] = None
        self._scaler_y: Optional[StandardScaler] = None
        self._train_prompt_ids: List[str] = []
        self._test_prompt_ids: List[str] = []
        self._train_mean: Optional[float] = None
        self._hidden_dim: Optional[int] = None
        self._prompt_dim: Optional[int] = None

    def _build_pipeline(self, prompt_dim: int, hidden_dim: int) -> SkPipeline:
        n_components = min(self.pca_components, hidden_dim - 1)
        if n_components < 1:
            n_components = 1

        steps = [
            ("pca_hidden", PCA(n_components=n_components, random_state=self.random_state)),
        ]
        return SkPipeline(steps)

    def fit(
        self,
        rows: List[FusionRow],
        train_ids: List[str],
        test_ids: List[str],
    ) -> "FusionPredictor":
        """Fit the model on training data only.

        PCA is fitted on training hidden states only. The final Ridge model
        operates on the concatenated [prompt_features (5-dim) + PCA_hidden (50-dim)]
        = 55-dim feature vector, standardized on training data only.
        """
        train_rows = [r for r in rows if r.prompt_id in set(train_ids)]
        train_prompt, train_hidden = build_fusion_matrix(train_rows)
        train_y = np.array([r.risk for r in train_rows], dtype=np.float32)

        self._train_mean = float(np.mean(train_y))
        self._prompt_dim = train_prompt.shape[1]
        self._hidden_dim = train_hidden.shape[1]

        # Fit PCA on training hidden states only.
        n_components = min(self.pca_components, train_hidden.shape[1] - 1, train_hidden.shape[0] - 1)
        if n_components < 1:
            n_components = 1
        pca = PCA(n_components=n_components,
                  random_state=self.random_state)
        train_hidden_pca = pca.fit_transform(train_hidden)

        # Concatenate prompt features + PCA hidden states.
        train_X = np.hstack([train_prompt, train_hidden_pca.astype(np.float32)])

        # Standardize combined features on training data only.
        self._scaler_X = StandardScaler()
        train_X_scaled = self._scaler_X.fit_transform(train_X)

        # Fit Ridge on standardized combined features.
        self._ridge = Ridge(alpha=self.alpha, random_state=self.random_state)
        # StandardScaler can produce NaNs if a feature has zero variance;
        # replace NaNs with 0 for ridge input.
        train_X_scaled = np.nan_to_num(train_X_scaled, nan=0.0)
        self._ridge.fit(train_X_scaled, train_y)

        self._pca = pca
        self._train_prompt_ids = list(train_ids)
        self._test_prompt_ids = list(test_ids)
        return self

    def _prepare_features(self, rows: List[FusionRow]) -> Tuple[np.ndarray, np.ndarray]:
        prompt_features, hidden_states = build_fusion_matrix(rows)
        hidden_pca = self._pca.transform(hidden_states)
        hidden_pca = np.nan_to_num(hidden_pca, nan=0.0)
        X = np.hstack([prompt_features, hidden_pca.astype(np.float32)])
        X_scaled = self._scaler_X.transform(X)
        X_scaled = np.nan_to_num(X_scaled, nan=0.0)
        return X_scaled, np.array([r.risk for r in rows])

    def predict(self, rows: List[FusionRow]) -> np.ndarray:
        """Predict risk for fusion rows."""
        if self._ridge is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        X, _ = self._prepare_features(rows)
        return self._ridge.predict(X)

    def evaluate(self, rows: List[FusionRow]) -> MetricsResult:
        """Evaluate on the held-out test set."""
        test_rows = [r for r in rows if r.prompt_id in set(self._test_prompt_ids)]
        X, test_y = self._prepare_features(test_rows)
        preds = self._ridge.predict(X)

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
        """Persist the fitted model, PCA, scaler, and metadata."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        model_path = output_dir / "fusion_model.joblib"
        artifacts = {
            "pca": self._pca,
            "scaler_X": self._scaler_X,
            "ridge": self._ridge,
        }
        joblib.dump(artifacts, model_path)

        metadata = {
            "model_type": "FusionPredictor",
            "alpha": self.alpha,
            "pca_components": self.pca_components,
            "hidden_dimensions": self._hidden_dim,
            "prompt_dimensions": self._prompt_dim,
            "combined_feature_dims": self._prompt_dim + self.pca_components,
            "random_state": self.random_state,
            "train_prompt_ids": self._train_prompt_ids,
            "test_prompt_ids": self._test_prompt_ids,
            "train_mean": self._train_mean,
            "uses_prompt_features": True,
            "uses_hidden_states": True,
            "hidden_state_representation": "last_layer_mean_pooled (PCA-reduced)",
            "fusion_strategy": "concatenate [prompt_features (5-dim) + PCA_hidden (50-dim)] -> StandardScaler -> Ridge",
            "training_only_preprocessing": True,
            "target_variable": "empirical_hallucination_risk",
            "risk_formula": "UNRELIABLE generations / total SUCCESS evaluations",
        }
        meta_path = output_dir / "metadata.json"
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(metadata, fh, indent=2)

        return model_path

    @classmethod
    def load(cls, model_path: Path, metadata_path: Optional[Path] = None) -> "FusionPredictor":
        """Load a persisted model and metadata."""
        model_path = Path(model_path)
        artifacts = joblib.load(model_path)

        predictor = cls()
        predictor._pca = artifacts["pca"]
        predictor._scaler_X = artifacts["scaler_X"]
        predictor._ridge = artifacts["ridge"]

        if metadata_path is not None:
            metadata_path = Path(metadata_path)
            if metadata_path.exists():
                with open(metadata_path, "r", encoding="utf-8") as fh:
                    meta = json.load(fh)
                predictor._train_prompt_ids = meta.get("train_prompt_ids", [])
                predictor._test_prompt_ids = meta.get("test_prompt_ids", [])
                predictor._train_mean = meta.get("train_mean")
                predictor._hidden_dim = meta.get("hidden_dimensions")
                predictor._prompt_dim = meta.get("prompt_dimensions")

        return predictor

    def get_feature_names(self) -> List[str]:
        """Return descriptive names for the fused feature representation."""
        names = list(FEATURE_COLUMNS)
        names.extend(["hidden_state_pca_{0}".format(i) for i in range(self.pca_components)])
        return names


def run_fusion_baseline(
    features_path: Path = DEFAULT_FEATURES_PATH,
    targets_path: Path = DEFAULT_TARGETS_PATH,
    hidden_states_dir: Path = DEFAULT_HIDDEN_STATES_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    test_size: float = 0.3,
    random_state: int = DEFAULT_RANDOM_STATE,
    train_ids: Optional[List[str]] = None,
    test_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """End-to-end fusion baseline training and evaluation.

    If train_ids/test_ids are provided (from a shared split), they are used
    to ensure all baselines are evaluated on the same prompt partition.
    """
    if not _HAS_SKLEARN:
        return {
            "status": "sklearn_unavailable",
            "model_type": "fusion",
        }

    rows, all_ds_indices, missing = align_feature_hidden_fusion(
        features_path, targets_path, hidden_states_dir
    )

    stats = {
        "model_type": "fusion",
        "prompts_aligned": len(rows),
        "prompts_missing_alignment": len(missing),
        "sufficient_for_training": len(rows) >= 5,
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
            "Only {0} prompts with features + hidden states + valid targets; "
            "minimum of 5 required.".format(len(rows))
        )
        return result

    prompt_ids = [r.prompt_id for r in rows]

    if train_ids is None or test_ids is None:
        train_ids, test_ids = train_test_split_by_prompt(
            prompt_ids, test_size=test_size, random_state=random_state
        )

    predictor = FusionPredictor(
        alpha=RIDGE_ALPHA,
        pca_components=PCA_N_COMPONENTS,
        random_state=random_state,
    )
    predictor.fit(rows, train_ids, test_ids)
    metrics = predictor.evaluate(rows)

    model_path = predictor.save(output_dir)

    predictions = {
        "model_type": "fusion",
        "test_prompt_ids": metrics.test_prompt_ids,
        "test_targets": metrics.test_targets,
        "predictions": metrics.predictions,
        "baseline_predictions": metrics.baseline_predictions,
        "train_mean": metrics.train_mean,
        "fusion_strategy": "prompt_features + hidden_state_pca",
    }
    pred_path = Path(output_dir) / "fusion_predictions.json"
    with open(pred_path, "w", encoding="utf-8") as fh:
        json.dump(predictions, fh, indent=2)

    result["status"] = "trained"
    result["split_info"] = {
        "train_prompt_ids": metrics.train_prompt_ids,
        "test_prompt_ids": metrics.test_prompt_ids,
        "train_prompt_count": len(metrics.train_prompt_ids),
        "test_prompt_count": len(metrics.test_prompt_ids),
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
        "prompt_dim": predictor._prompt_dim,
        "hidden_dim": predictor._hidden_dim,
        "combined_dim": predictor._prompt_dim + PCA_N_COMPONENTS,
    }
    result["model_path"] = str(model_path)
    result["predictions_path"] = str(pred_path)
    return result
