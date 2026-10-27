"""Pre-generation baseline inference. No target model or evaluator is loaded."""
from dataclasses import asdict, dataclass
import hashlib
import json
import logging
import math
from pathlib import Path
import statistics
from typing import Mapping

MODEL = 'Qwen/Qwen3-1.7B'
REVISION = '70d244cc86ccca08cf5af4e1e306ecf908b1ad5e'
PIPELINE = 'constant-no-features-v1'
GENERATION_CONFIG = dict(device='cpu', thinking_mode=False, do_sample=True,
                         temperature=.7, top_p=.8, top_k=20, max_new_tokens=512)
LOGGER = logging.getLogger(__name__)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                     separators=(',', ':')).encode()).hexdigest()


def _risk(value):
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError('Risk must be a finite number in [0, 1]')


@dataclass(frozen=True)
class Artifact:
    schema_version: int
    feature_pipeline_version: str
    target_model: str
    model_revision: str
    generation_config: dict
    statistic: str
    value: float
    n_train: int
    training_ids_sha256: str
    source_sha256: dict
    predictor_id: str

    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError('Unsupported artifact schema')
        if self.feature_pipeline_version != PIPELINE or self.target_model != MODEL or self.model_revision != REVISION:
            raise ValueError('Incompatible feature pipeline or target model')
        _validate_config(self.generation_config)
        if self.statistic not in ('mean', 'median'):
            raise ValueError('Unsupported baseline statistic')
        _risk(self.value)
        if type(self.n_train) is not int or self.n_train < 1:
            raise ValueError('Invalid training count')
        hashes = [self.training_ids_sha256]
        if not isinstance(self.source_sha256, dict) or not self.source_sha256:
            raise ValueError('Source provenance is required')
        hashes.extend(self.source_sha256.values())
        if any(not isinstance(h, str) or len(h) != 64 or any(c not in '0123456789abcdef' for c in h) for h in hashes):
            raise ValueError('Invalid provenance hash')
        body = self.to_dict()
        body.pop('predictor_id')
        if self.predictor_id != _digest(body):
            raise ValueError('Artifact integrity mismatch')

    @classmethod
    def fit(cls, targets, prompt_ids, statistic, source_sha256):
        y, ids = list(targets), list(prompt_ids)
        if not y or len(y) != len(ids) or any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
            raise ValueError('Training IDs must be unique and aligned with targets')
        for value in y:
            _risk(value)
        if statistic not in ('mean', 'median'):
            raise ValueError('Unsupported baseline statistic')
        body = dict(schema_version=1, feature_pipeline_version=PIPELINE,
                    target_model=MODEL, model_revision=REVISION,
                    generation_config=dict(GENERATION_CONFIG), statistic=statistic,
                    value=float(statistics.mean(y) if statistic == 'mean' else statistics.median(y)),
                    n_train=len(y), training_ids_sha256=_digest(sorted(ids)),
                    source_sha256=dict(source_sha256))
        return cls(**body, predictor_id=_digest(body))

    def to_dict(self):
        return asdict(self)

    def save(self, path):
        # Exclusive creation preserves existing model versions.
        self.__post_init__()
        with Path(path).open('x', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, allow_nan=False)

    @classmethod
    def load(cls, path):
        def unique(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError('Duplicate JSON key')
                result[key] = value
            return result
        try:
            return cls(**json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=unique))
        except (TypeError, KeyError) as exc:
            raise ValueError('Malformed prediction artifact') from exc


def _validate_config(config):
    if not isinstance(config, Mapping) or set(config) != set(GENERATION_CONFIG):
        raise ValueError('Exact generation configuration is required')
    for key, expected in GENERATION_CONFIG.items():
        if type(config[key]) is not type(expected) or config[key] != expected:
            raise ValueError('Unsupported generation configuration: ' + key)


@dataclass(frozen=True)
class Prediction:
    estimated_risk: float
    predictor_id: str
    target_model: str
    model_revision: str
    feature_pipeline_version: str
    baseline_statistic: str
    calibrated: bool = False
    uncertainty: None = None
    warnings: tuple = (
        'Cohort baseline; no prompt-specific discrimination or correctness guarantee.',
        'Evaluator-derived SQuAD risk; external validity and calibration are unverified.',
        'Individual uncertainty is unsupported; zero median does not imply a safe answer.',
    )

    def to_dict(self):
        return asdict(self)


def predict_risk(prompt, target_model, model_revision, config, *, artifact):
    """Accept rendered text or exactly {question, context}; reject enriched records.

    Constants use no prompt features. Training and inference have the same empty
    feature schema. Callers must explicitly load a baseline; no implicit fallback.
    """
    if target_model != MODEL or model_revision != REVISION:
        raise ValueError('Unsupported target model or revision')
    _validate_config(config)
    if isinstance(prompt, str):
        parts = [prompt]
    elif isinstance(prompt, Mapping) and set(prompt) == {'question', 'context'}:
        parts = list(prompt.values())
    else:
        raise ValueError('Prompt must be text or exactly question/context; metadata is forbidden')
    if any(not isinstance(p, str) or not p.strip() or len(p) > 100_000 for p in parts):
        raise ValueError('Prompt fields must contain 1 to 100000 characters')
    if not isinstance(artifact, Artifact):
        raise ValueError('An explicitly loaded prediction artifact is required')
    artifact.__post_init__()  # also detects accidental mutation of nested dictionaries
    LOGGER.info('lux_prediction', extra={'predictor_id': artifact.predictor_id,
                                        'pipeline_version': PIPELINE})
    return Prediction(artifact.value, artifact.predictor_id, MODEL, REVISION,
                      PIPELINE, artifact.statistic)


@dataclass(frozen=True)
class Policy:
    verify_at: float | None = None
    escalate_at: float | None = None
    allow_uncalibrated: bool = False

    def __post_init__(self):
        if type(self.allow_uncalibrated) is not bool:
            raise ValueError('allow_uncalibrated must be boolean')
        if self.verify_at is None and self.escalate_at is None:
            return
        _risk(self.verify_at)
        _risk(self.escalate_at)
        if self.verify_at >= self.escalate_at:
            raise ValueError('Thresholds must be strictly ordered')

    def route(self, prediction):
        if not isinstance(prediction, Prediction) or self.verify_at is None:
            return 'escalate'
        if not prediction.calibrated and not self.allow_uncalibrated:
            return 'escalate'
        try:
            _risk(prediction.estimated_risk)
        except ValueError:
            return 'escalate'
        if prediction.estimated_risk >= self.escalate_at:
            return 'escalate'
        return 'verify' if prediction.estimated_risk >= self.verify_at else 'allow'
