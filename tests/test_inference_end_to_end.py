"""Public API integration with shipped artifacts; no generation or training."""
from dataclasses import replace
import json
from pathlib import Path

import pytest

from src.predictor.inference import (
    Artifact, GENERATION_CONFIG, MODEL, PIPELINE, REVISION, Policy, predict_risk,
)

ARTIFACTS = Path(__file__).resolve().parents[1] / 'docs/engineering'


@pytest.fixture
def baseline():
    return Artifact.load(ARTIFACTS / 'baseline_mean_v2.json')


def predict(prompt, baseline, **overrides):
    args = dict(target_model=MODEL, model_revision=REVISION,
                config=dict(GENERATION_CONFIG), artifact=baseline)
    return predict_risk(prompt, **{**args, **overrides})


@pytest.mark.parametrize('prompt', ['Why?', {'question': 'क्यों?', 'context': 'A passage.'}, 'x' * 100000],
                         ids=['text', 'structured_unicode', 'maximum_length'])
def test_shipped_artifact_complete_path(prompt, baseline):
    result = predict(prompt, baseline)
    output = json.loads(json.dumps(result.to_dict(), allow_nan=False))
    assert set(output) == {'estimated_risk', 'predictor_id', 'target_model', 'model_revision',
                           'feature_pipeline_version', 'baseline_statistic', 'calibrated',
                           'uncertainty', 'warnings'}
    assert type(output['estimated_risk']) is float
    assert 0 <= output['estimated_risk'] <= 1
    assert output['estimated_risk'] == pytest.approx(.20864197530864198)
    assert output['predictor_id'] == baseline.predictor_id
    assert output['target_model'] == MODEL and output['model_revision'] == REVISION
    assert output['feature_pipeline_version'] == PIPELINE
    assert output['baseline_statistic'] == 'mean'
    assert output['calibrated'] is False and output['uncertainty'] is None
    assert output['warnings'] and all(isinstance(w, str) for w in output['warnings'])
    assert Policy().route(result) == 'escalate'


@pytest.mark.parametrize('content', ['{', 'null', '[]', '{}', '{"schema_version":1,"schema_version":1}', '\ufffd'])
def test_corrupt_artifacts_raise_without_prediction(tmp_path, content):
    path = tmp_path / 'bad.json'
    path.write_text(content, encoding='utf-8')
    with pytest.raises(ValueError):
        Artifact.load(path)
    assert Policy().route(None) == 'escalate'


@pytest.mark.parametrize('field,value', [('schema_version', 0), ('schema_version', True),
    ('feature_pipeline_version', 'hidden-state-v1'), ('model_revision', 'other'),
    ('target_model', 'other'), ('value', -1), ('value', 2), ('value', float('inf')),
    ('predictor_id', '0' * 64), ('n_train', 0)])
def test_incompatible_or_tampered_artifacts(tmp_path, baseline, field, value):
    payload = baseline.to_dict()
    payload[field] = value
    path = tmp_path / 'incompatible.json'
    path.write_text(json.dumps(payload), encoding='utf-8')
    with pytest.raises(ValueError):
        Artifact.load(path)


def test_artifact_schema_rejects_missing_extra_and_invalid_encoding(tmp_path, baseline):
    path = tmp_path / 'malformed.json'
    payload = baseline.to_dict()
    payload.pop('value')
    for content in [json.dumps(payload).encode(),
                    json.dumps({**baseline.to_dict(), 'hidden_states': []}).encode(), b'\xff\xfe']:
        path.write_bytes(content)
        with pytest.raises(ValueError):
            Artifact.load(path)


def test_operational_sample_covers_documented_sizes(baseline):
    from docs.engineering.measure_baseline_inference import sample
    cases = sample()
    assert len(cases) == 12 and len({name for name, _ in cases}) == 12
    assert [len(prompt) for _, prompt in cases if isinstance(prompt, str)] == [32, 256, 1024, 4096, 16384, 65536]
    assert all(set(prompt) == {'question', 'context'} for _, prompt in cases if isinstance(prompt, dict))
    for _, prompt in cases:
        result = predict(prompt, baseline)
        assert result.estimated_risk == pytest.approx(.20864197530864198)
        assert Policy().route(result) == 'escalate'


def test_missing_artifact_and_explicit_observable_fallback(tmp_path, caplog):
    import logging
    with pytest.raises(FileNotFoundError):
        Artifact.load(tmp_path / 'unavailable.json')
    assert Policy().route(None) == 'escalate'
    # Application explicitly elects a cohort baseline after a preferred artifact fails.
    chosen = Artifact.load(ARTIFACTS / 'baseline_mean_v2.json')
    with caplog.at_level(logging.INFO, logger='src.predictor.inference'):
        result = predict('Question?', chosen)
    assert result.baseline_statistic == 'mean'
    assert result.feature_pipeline_version == 'constant-no-features-v1'
    assert caplog.records[-1].predictor_id == chosen.predictor_id
    median = Artifact.load(ARTIFACTS / 'baseline_median_v2.json')
    assert predict('Question?', median).estimated_risk == 0
    assert median.predictor_id != chosen.predictor_id
    assert Policy().route(predict('Question?', median)) == 'escalate'


@pytest.mark.parametrize('extra', ['generated_answer', 'generation_label', 'evaluation_status',
    'claim_evaluations', 'empirical_hallucination_risk', 'answerability', 'hidden_states',
    'hidden_state_ref', 'prompt_metrics'])
def test_no_post_generation_or_hidden_feature_injection(baseline, extra):
    with pytest.raises(ValueError):
        predict({'question': 'Why?', 'context': 'Text', extra: [0] * 3}, baseline)


@pytest.mark.parametrize('prompt', [None, 3, [], b'question', '', '\t\n',
    {'question': None, 'context': 'text'}, {'question': 'x', 'context': {}},
    {'question': 'x'}, {'question': 'x', 'context': 'x' * 100001}])
def test_bad_inputs_never_return_low_risk(prompt, baseline):
    with pytest.raises(ValueError):
        predict(prompt, baseline)


@pytest.mark.parametrize('override', [{'target_model': 'unsupported'}, {'model_revision': 'main'},
    {'config': {}}, {'config': None}, {'config': {**GENERATION_CONFIG, 'generated_answer': 'x'}},
    {'artifact': None}])
def test_unsupported_request(override, baseline):
    with pytest.raises(ValueError):
        predict('Question?', baseline, **override)


def test_repeat_requests_are_deterministic_without_mutation(baseline):
    initial = baseline.to_dict()
    first = predict('Question?', baseline)
    for _ in range(500):
        result = predict('Question?', baseline)
        assert result == first
        assert Policy().route(result) == 'escalate'
    assert baseline.to_dict() == initial
    for value in [float('nan'), float('inf'), -.1, 1.1, True]:
        # Reuse existing test-only policy values, not deployment recommendations.
        assert Policy(.2, .7, allow_uncalibrated=True).route(replace(first, estimated_risk=value)) == 'escalate'
