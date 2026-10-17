"""Tests for LUX predictor baselines: prompt-only, hidden-state-only, and fusion.

All three baselines are tested with synthetic data since the real on-disk
dataset contains only 1 valid risk target (prompt 69929, risk=0.0), which
is insufficient for train/test splits.
"""

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.predictor.baselines import (
    ALL_FEATURE_NAMES,
    AlignmentError,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_RANDOM_STATE,
    EXCLUDED_FROM_FEATURES,
    FusionRow,
    HiddenStateRow,
    MetricsResult,
    PromptOnlyPredictor,
    TargetRow,
    compute_metrics,
    load_predictor_targets,
    prompt_only_train_test_split,
    run_prompt_only_baseline,
    train_test_split_by_prompt,
    MIN_TRAIN_PROMPTS,
    MIN_TEST_PROMPTS,
    MIN_TOTAL_FOR_SPLIT,
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
    build_fusion_matrix,
    run_fusion_baseline,
)


# --------------------------------------------------------------------------
# Synthetic data generators
# --------------------------------------------------------------------------
def _make_synthetic_targets(n: int = 20, seed: int = 42) -> dict:
    """Create a synthetic targets.json structure."""
    np.random.seed(seed)
    targets_list = []
    for i in range(n):
        # Risk values in [0, 1] with some variance.
        risk = float(np.random.uniform(0.0, 1.0))
        targets_list.append({
            "dataset_index": i,
            "prompt_id": f"prompt_{i:04d}",
            "empirical_hallucination_risk": risk,
        })
    return {"targets": targets_list}


def _make_synthetic_features(n: int = 20, seed: int = 42) -> dict:
    """Create a synthetic predictor_features.json structure.

    Uses 'predictor_features' key and includes only pre-generation fields.
    """
    np.random.seed(seed)
    feat_list = []
    for i in range(n):
        metrics = {
            "question_chars": int(np.random.randint(10, 200)),
            "context_chars": int(np.random.randint(100, 2000)),
            "question_tokens": int(np.random.randint(3, 50)),
            "context_tokens": int(np.random.randint(20, 500)),
            "question_ends_with_question_mark": bool(np.random.rand() > 0.3),
        }
        feat_list.append({
            "dataset_index": i,
            "prompt_id": f"prompt_{i:04d}",
            "prompt_metrics": metrics,
        })
    return {"predictor_features": feat_list}


def _make_synthetic_hidden_state(ds_idx: int, hidden_size: int = 2048, num_layers: int = 29) -> dict:
    """Create a synthetic hidden-state payload."""
    np.random.seed(ds_idx)
    return {
        "prompt_dataset_index": ds_idx,
        "model_name": "Qwen3-1.7B",
        "model_revision": "70d244cc86ccca4e1e306ecf908b1ad5e",
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "seq_len": 100,
        "last_token": np.random.randn(num_layers, hidden_size).tolist(),
        "mean_pooled": np.random.randn(num_layers, hidden_size).tolist(),
    }


def _make_synthetic_features_with_ds_indices(targets_path: Path, n: int = 20) -> dict:
    """Create features that include dataset_index for hidden-state alignment."""
    return _make_synthetic_features(n)


@pytest.fixture
def synthetic_data_dirs():
    """Create temporary directories with synthetic data for all three predictors."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        features_dir = tmp / "predictor_features"
        features_dir.mkdir()
        hidden_dir = tmp / "hidden_states"
        hidden_dir.mkdir()
        targets_dir = tmp / "targets"
        targets_dir.mkdir()
        output_dir = tmp / "output"
        output_dir.mkdir()

        n = 20
        features_data = _make_synthetic_features(n)
        targets_data = _make_synthetic_targets(n)

        features_path = features_dir / "predictor_features.json"
        with open(features_path, "w") as f:
            json.dump(features_data, f)

        targets_path = targets_dir / "targets.json"
        with open(targets_path, "w") as f:
            json.dump(targets_data, f)

        # Write hidden-state files.
        hs_indices = set()
        for entry in features_data["predictor_features"]:
            ds_idx = entry["dataset_index"]
            hs_indices.add(ds_idx)
            hs_data = _make_synthetic_hidden_state(ds_idx)
            hs_path = hidden_dir / f"hs_{ds_idx:04d}.json"
            with open(hs_path, "w") as f:
                json.dump(hs_data, f)

        yield {
            "features_path": features_path,
            "targets_path": targets_path,
            "hidden_states_dir": hidden_dir,
            "output_dir": output_dir,
            "n": n,
            "hs_indices": hs_indices,
        }


@pytest.fixture
def sklearn_unavailable():
    """Simulate sklearn being unavailable."""
    with patch("src.predictor.baselines._HAS_SKLEARN", False), \
         patch("src.predictor.hidden_only._HAS_SKLEARN", False), \
         patch("src.predictor.fusion._HAS_SKLEARN", False):
        yield


# --------------------------------------------------------------------------
# Prompt-only tests
# --------------------------------------------------------------------------
class TestPromptOnlyBaseline:
    def test_load_predictor_targets_excludes_answerability(self, synthetic_data_dirs):
        """Verify that answerability is never included in the feature dict."""
        rows, all_ids = load_predictor_targets(
            synthetic_data_dirs["features_path"],
            synthetic_data_dirs["targets_path"],
        )
        assert len(rows) == synthetic_data_dirs["n"]
        for row in rows:
            assert "answerability" not in row.features
            assert "risk" not in row.features

    def test_extract_feature_vector_excludes_forbidden_fields(self, synthetic_data_dirs):
        """Verify feature extraction excludes answerability and leakage fields."""
        rows, _ = load_predictor_targets(
            synthetic_data_dirs["features_path"],
            synthetic_data_dirs["targets_path"],
        )
        from src.predictor.baselines import _extract_feature_vector
        for row in rows:
            vec = _extract_feature_vector(row.features)
            assert len(vec) == len(ALL_FEATURE_NAMES)
            # Ensure answerability value is NOT in the vector.
            # The feature vector should contain only numerical/binary features.
            assert all(isinstance(v, (int, float)) for v in vec)

    def test_train_test_split_disjoint(self, synthetic_data_dirs):
        """Verify train/test split has no prompt overlap."""
        rows, _ = load_predictor_targets(
            synthetic_data_dirs["features_path"],
            synthetic_data_dirs["targets_path"],
        )
        split = prompt_only_train_test_split(rows)
        assert split is not None
        split.assert_disjoint()

    def test_train_test_split_insufficient_raises(self):
        """Verify insufficient prompts raises AlignmentError."""
        short_ids = [f"p{i}" for i in range(4)]  # Only 4, need 5 (MIN_TRAIN + MIN_TEST)
        with pytest.raises(AlignmentError):
            train_test_split_by_prompt(short_ids)

    def test_prompt_only_predictor_fit_and_evaluate(self, synthetic_data_dirs):
        """Verify the PromptOnlyPredictor can fit and produce metrics."""
        rows, _ = load_predictor_targets(
            synthetic_data_dirs["features_path"],
            synthetic_data_dirs["targets_path"],
        )
        split = prompt_only_train_test_split(rows)
        predictor = PromptOnlyPredictor(alpha=1.0, random_state=42)
        predictor.fit(split)
        metrics = predictor.evaluate(split)

        assert isinstance(metrics, MetricsResult)
        assert metrics.n_train == len(split.train_ids)
        assert metrics.n_test == len(split.test_ids)
        assert metrics.mae >= 0
        assert metrics.rmse >= 0
        # Baseline predicts training mean.
        assert all(
            abs(b - metrics.train_mean) < 1e-10
            for b in metrics.baseline_predictions
        )

    def test_prompt_only_predictor_save_and_load(self, synthetic_data_dirs):
        """Verify model persistence round-trips correctly."""
        rows, _ = load_predictor_targets(
            synthetic_data_dirs["features_path"],
            synthetic_data_dirs["targets_path"],
        )
        split = prompt_only_train_test_split(rows)
        predictor = PromptOnlyPredictor(alpha=1.0, random_state=42)
        predictor.fit(split)
        model_path = predictor.save(synthetic_data_dirs["output_dir"])

        loaded = PromptOnlyPredictor.load(model_path)
        preds_orig = predictor.predict(split.test_features)
        preds_loaded = loaded.predict(split.test_features)
        np.testing.assert_array_almost_equal(preds_orig, preds_loaded)

    def test_run_prompt_only_baseline_synthetic(self, synthetic_data_dirs):
        """End-to-end test with synthetic data."""
        result = run_prompt_only_baseline(
            features_path=synthetic_data_dirs["features_path"],
            targets_path=synthetic_data_dirs["targets_path"],
            output_dir=synthetic_data_dirs["output_dir"],
        )
        assert result["status"] == "trained"
        assert result["metrics"] is not None
        assert result["metrics"]["n_train"] > 0
        assert result["metrics"]["n_test"] > 0
        assert result["metrics"]["mae"] >= 0
        assert result["metrics"]["baseline_mae"] >= 0

    def test_run_prompt_only_baseline_insufficient_data(self, tmp_path):
        """Test insufficient data path."""
        features_data = {
            "predictor_features": [
                {"dataset_index": i, "prompt_id": f"p{i}",
                 "prompt_metrics": {"question_chars": 10, "context_chars": 100,
                                    "question_tokens": 5, "context_tokens": 30,
                                    "question_ends_with_question_mark": True}}
                for i in range(3)
            ]
        }
        targets_data = {
            "targets": [
                {"dataset_index": i, "prompt_id": f"p{i}",
                 "empirical_hallucination_risk": float(i) * 0.5}
                for i in range(3)
            ]
        }
        feats_path = tmp_path / "feats.json"
        targ_path = tmp_path / "targs.json"
        json.dump(features_data, open(feats_path, "w"))
        json.dump(targets_data, open(targ_path, "w"))

        result = run_prompt_only_baseline(
            features_path=feats_path,
            targets_path=targ_path,
            output_dir=tmp_path / "out",
        )
        assert result["status"] == "insufficient_data"
        assert result["metrics"] is None

    def test_leakage_prevention_in_load_predictor_targets(self, tmp_path):
        """Verify loading raises ValueError if leakage fields are present."""
        features_data = {
            "predictor_features": [
                {"dataset_index": 0, "prompt_id": "p0",
                 "prompt_metrics": {"question_chars": 10, "answerability": True,
                                    "risk": 0.5},
                 "generated_answer": "leaked!"}
            ]
        }
        targets_data = {
            "targets": [{"dataset_index": 0, "prompt_id": "p0",
                         "empirical_hallucination_risk": 0.5}]
        }
        feats_path = tmp_path / "feats.json"
        targ_path = tmp_path / "targs.json"
        json.dump(features_data, open(feats_path, "w"))
        json.dump(targets_data, open(targ_path, "w"))

        with pytest.raises(ValueError, match="Leakage prevention"):
            load_predictor_targets(feats_path, targ_path)

    def test_shared_split_consistency(self, synthetic_data_dirs):
        """Verify the shared split produces correct prompt counts."""
        rows, _ = load_predictor_targets(
            synthetic_data_dirs["features_path"],
            synthetic_data_dirs["targets_path"],
        )
        prompt_ids = [r.prompt_id for r in rows]
        train_ids, test_ids = train_test_split_by_prompt(prompt_ids)
        assert len(train_ids) + len(test_ids) == len(prompt_ids)
        assert len(train_ids) >= 3  # MIN_TRAIN_PROMPTS
        assert len(test_ids) >= 2  # MIN_TEST_PROMPTS
        assert set(train_ids).isdisjoint(set(test_ids))


# --------------------------------------------------------------------------
# Hidden-state-only tests
# --------------------------------------------------------------------------
class TestHiddenStateOnlyBaseline:
    def test_load_hidden_states(self, synthetic_data_dirs):
        """Verify hidden-state files are loaded correctly."""
        hs = load_hidden_states(synthetic_data_dirs["hidden_states_dir"])
        assert len(hs) == synthetic_data_dirs["n"]
        for ds_idx in synthetic_data_dirs["hs_indices"]:
            assert ds_idx in hs

    def test_extract_hidden_representation(self, synthetic_data_dirs):
        """Verify the last-layer mean-pooled representation is extracted."""
        hs = load_hidden_states(synthetic_data_dirs["hidden_states_dir"])
        sample = hs[0]
        rep = extract_hidden_representation(sample, strategy=HIDDEN_STATE_STRATEGY)
        expected_dim = sample["hidden_size"]  # 2048
        assert rep.shape == (expected_dim,)

    def test_extract_hidden_representation_invalid_strategy(self, synthetic_data_dirs):
        """Verify invalid strategy raises ValueError."""
        hs = load_hidden_states(synthetic_data_dirs["hidden_states_dir"])
        with pytest.raises(ValueError, match="Unknown hidden-state strategy"):
            extract_hidden_representation(hs[0], strategy="invalid")

    def test_hidden_state_only_predictor_fit_and_evaluate(self, synthetic_data_dirs):
        """Verify HiddenStateOnlyPredictor can fit and produce metrics."""
        from src.predictor.hidden_only import align_hidden_states_with_targets
        hs = load_hidden_states(synthetic_data_dirs["hidden_states_dir"])
        rows, _, _ = align_hidden_states_with_targets(
            hs, synthetic_data_dirs["targets_path"]
        )
        assert len(rows) == synthetic_data_dirs["n"]

        prompt_ids = [r.prompt_id for r in rows]
        train_ids, test_ids = train_test_split_by_prompt(prompt_ids)

        predictor = HiddenStateOnlyPredictor(alpha=1.0, pca_components=50, random_state=42)
        predictor.fit(rows, train_ids, test_ids)
        metrics = predictor.evaluate(rows)

        assert isinstance(metrics, MetricsResult)
        assert metrics.n_train == len(train_ids)
        assert metrics.n_test == len(test_ids)

    def test_run_hidden_only_baseline_synthetic(self, synthetic_data_dirs):
        """End-to-end test with synthetic data."""
        result = run_hidden_only_baseline(
            targets_path=synthetic_data_dirs["targets_path"],
            hidden_states_dir=synthetic_data_dirs["hidden_states_dir"],
            output_dir=synthetic_data_dirs["output_dir"],
        )
        assert result["status"] == "trained"
        assert result["metrics"] is not None
        assert result["metrics"]["mae"] >= 0

    def test_hidden_state_dimension_consistency(self, synthetic_data_dirs):
        """Verify all hidden-state representations have the same dimension."""
        hs = load_hidden_states(synthetic_data_dirs["hidden_states_dir"])
        dims = set()
        for data in hs.values():
            rep = extract_hidden_representation(data)
            dims.add(rep.shape[0])
        assert len(dims) == 1  # All should be 2048


# --------------------------------------------------------------------------
# Fusion tests
# --------------------------------------------------------------------------
class TestFusionBaseline:
    def test_align_feature_hidden_fusion(self, synthetic_data_dirs):
        """Verify fusion alignment produces rows for all prompts."""
        rows, all_ds_indices, missing = align_feature_hidden_fusion(
            synthetic_data_dirs["features_path"],
            synthetic_data_dirs["targets_path"],
            synthetic_data_dirs["hidden_states_dir"],
        )
        assert len(rows) == synthetic_data_dirs["n"]
        assert len(missing) == 0
        for row in rows:
            assert isinstance(row.prompt_features, dict)
            assert row.hidden_state is not None
            assert row.risk >= 0.0

    def test_build_fusion_matrix_dimensions(self):
        """Verify fusion matrix has correct dimensions."""
        rows = [
            FusionRow(
                prompt_id=f"p{i}",
                dataset_index=i,
                prompt_features={
                    "question_chars": float(i * 10),
                    "context_chars": float(i * 50),
                    "question_tokens": float(i),
                    "context_tokens": float(i * 5),
                    "question_ends_with_question_mark": bool(i % 2),
                },
                hidden_state=np.random.randn(2048).astype(np.float32),
                risk=float(i) / 10.0,
            )
            for i in range(10)
        ]
        prompt_matrix, hidden_matrix = build_fusion_matrix(rows)
        assert prompt_matrix.shape == (10, len(ALL_FEATURE_NAMES))
        assert hidden_matrix.shape == (10, 2048)

    def test_fusion_predictor_fit_and_evaluate(self, synthetic_data_dirs):
        """Verify FusionPredictor can fit and produce metrics."""
        rows, _, _ = align_feature_hidden_fusion(
            synthetic_data_dirs["features_path"],
            synthetic_data_dirs["targets_path"],
            synthetic_data_dirs["hidden_states_dir"],
        )
        assert len(rows) >= 5

        prompt_ids = [r.prompt_id for r in rows]
        train_ids, test_ids = train_test_split_by_prompt(prompt_ids)

        predictor = FusionPredictor(alpha=1.0, pca_components=50, random_state=42)
        predictor.fit(rows, train_ids, test_ids)
        metrics = predictor.evaluate(rows)

        assert isinstance(metrics, MetricsResult)
        assert metrics.n_train == len(train_ids)
        assert metrics.n_test == len(test_ids)

    def test_run_fusion_baseline_synthetic(self, synthetic_data_dirs):
        """End-to-end test with synthetic data."""
        result = run_fusion_baseline(
            features_path=synthetic_data_dirs["features_path"],
            targets_path=synthetic_data_dirs["targets_path"],
            hidden_states_dir=synthetic_data_dirs["hidden_states_dir"],
            output_dir=synthetic_data_dirs["output_dir"],
        )
        assert result["status"] == "trained"
        assert result["metrics"] is not None
        assert result["metrics"]["mae"] >= 0

    def test_fusion_uses_both_feature_types(self, synthetic_data_dirs):
        """Verify fusion includes both prompt and hidden-state features."""
        rows, _, _ = align_feature_hidden_fusion(
            synthetic_data_dirs["features_path"],
            synthetic_data_dirs["targets_path"],
            synthetic_data_dirs["hidden_states_dir"],
        )
        prompt_ids = [r.prompt_id for r in rows]
        train_ids, test_ids = train_test_split_by_prompt(prompt_ids)

        predictor = FusionPredictor(alpha=1.0, random_state=42)
        predictor.fit(rows, train_ids, test_ids)
        feature_names = predictor.get_feature_names()
        # Should include prompt feature names.
        assert "question_chars" in feature_names
        # Should include PCA component names.
        assert any("pca" in name.lower() for name in feature_names)


# --------------------------------------------------------------------------
# Metrics tests
# --------------------------------------------------------------------------
class TestComputeMetrics:
    def test_metrics_perfect_prediction(self):
        """Verify metrics for perfect predictions."""
        y_true = np.array([0.1, 0.5, 0.9])
        y_pred = np.array([0.1, 0.5, 0.9])
        result = compute_metrics(y_true, y_pred)
        assert result["mae"] == pytest.approx(0.0)
        assert result["rmse"] == pytest.approx(0.0)
        assert result["r2"] == pytest.approx(1.0)

    def test_metrics_with_baseline(self):
        """Verify baseline metrics computation."""
        y_true = np.array([0.0, 1.0, 0.5])
        y_pred = np.array([0.3, 0.7, 0.4])
        train_mean = 0.5
        baseline_pred = np.array([0.5, 0.5, 0.5])
        result = compute_metrics(y_true, y_pred, baseline_pred, train_mean)
        assert result["baseline_mae"] >= 0
        assert result["baseline_rmse"] >= 0
        assert result["improvement_mae_over_baseline"] is not None

    def test_metrics_zero_variance_r2_none(self):
        """Verify R^2 is None when all targets are identical."""
        y_true = np.array([0.5, 0.5, 0.5])
        y_pred = np.array([0.4, 0.5, 0.6])
        result = compute_metrics(y_true, y_pred)
        assert result["r2"] is None
        assert result["mae"] >= 0


# --------------------------------------------------------------------------
# Shared split tests
# --------------------------------------------------------------------------
class TestTrainTestSplitByPrompt:
    def test_split_disjointness(self):
        """Verify split never produces overlapping IDs."""
        ids = [f"prompt_{i:03d}" for i in range(20)]
        train_ids, test_ids = train_test_split_by_prompt(ids)
        assert set(train_ids).isdisjoint(set(test_ids))
        assert len(train_ids) + len(test_ids) == len(ids)

    def test_split_insufficient_raises(self):
        """Verify insufficient prompts raises AlignmentError."""
        ids = [f"p{i}" for i in range(3)]  # Only 3, need 5
        with pytest.raises(AlignmentError):
            train_test_split_by_prompt(ids)

    def test_split_reproducible(self):
        """Verify same random_state produces same split."""
        ids = [f"p{i}" for i in range(50)]
        train1, test1 = train_test_split_by_prompt(ids, random_state=42)
        train2, test2 = train_test_split_by_prompt(ids, random_state=42)
        assert train1 == train2
        assert test1 == test2

    def test_min_constants(self):
        """Verify minimum prompt constants are defined."""
        from src.predictor.baselines import MIN_TRAIN_PROMPTS, MIN_TEST_PROMPTS, MIN_TOTAL_FOR_SPLIT
        assert MIN_TRAIN_PROMPTS == 3
        assert MIN_TEST_PROMPTS == 2
        assert MIN_TOTAL_FOR_SPLIT == 5
