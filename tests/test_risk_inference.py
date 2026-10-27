import json

import pytest

from src.predictor.inference import (
    Artifact, GENERATION_CONFIG, MODEL, REVISION, Policy, predict_risk,
)


def artifact():
    return Artifact.fit([0, .2, 1], ['a', 'b', 'c'], 'mean', {'snapshot': 'a' * 64})


def test_roundtrip_and_prediction(tmp_path):
    a = artifact()
    path = tmp_path / 'baseline.json'
    a.save(path)
    b = Artifact.load(path)
    r = predict_risk({'question': 'Why?', 'context': 'Some context'}, MODEL, REVISION,
                     dict(GENERATION_CONFIG), artifact=b)
    assert r.estimated_risk == pytest.approx(.4)
    assert r.uncertainty is None and not r.calibrated
    assert r.to_dict()['predictor_id'] == a.predictor_id
    with pytest.raises(FileExistsError):
        a.save(path)


@pytest.mark.parametrize('prompt', ['', '   ', None, {}, {'question': 'x', 'context': 'y', 'generated_answer': 'z'},
    {'question': 'x', 'context': 'y', 'labels': []}, {'question': 'x', 'context': ''}])
def test_invalid_or_leaking_prompt(prompt):
    with pytest.raises(ValueError):
        predict_risk(prompt, MODEL, REVISION, GENERATION_CONFIG, artifact=artifact())


@pytest.mark.parametrize('change', [{'temperature': .9}, {'hidden_states': []}, {'thinking_mode': True}, {'top_k': True}])
def test_unsupported_config(change):
    with pytest.raises(ValueError):
        predict_risk('Question?', MODEL, REVISION, {**GENERATION_CONFIG, **change}, artifact=artifact())


def test_identity_and_artifact_validation(tmp_path):
    with pytest.raises(ValueError):
        predict_risk('Question?', MODEL, 'wrong', GENERATION_CONFIG, artifact=artifact())
    p = tmp_path / 'bad.json'
    a = artifact().to_dict()
    for key, value in [('feature_pipeline_version', 'wrong'), ('value', float('nan')), ('schema_version', 2)]:
        p.write_text(json.dumps({**a, key: value}))
        with pytest.raises(ValueError):
            Artifact.load(p)


@pytest.mark.parametrize('y,ids', [([float('nan')], ['a']), ([1.1], ['a']), ([0, 1], ['a', 'a']), ([], []), ([0], [])])
def test_invalid_training(y, ids):
    with pytest.raises(ValueError):
        Artifact.fit(y, ids, 'mean', {'snapshot': 'a' * 64})


def test_policy_has_no_implicit_thresholds():
    r = predict_risk('Question?', MODEL, REVISION, GENERATION_CONFIG, artifact=artifact())
    assert Policy().route(r) == 'escalate'
    assert Policy(.2, .7, allow_uncalibrated=True).route(r) == 'verify'
    assert Policy(.5, .7, allow_uncalibrated=True).route(r) == 'allow'
    assert Policy(.2, .3, allow_uncalibrated=True).route(r) == 'escalate'
    assert Policy(.5, .7).route(r) == 'escalate'
    assert Policy(.5, .7, allow_uncalibrated=True).route(None) == 'escalate'
    with pytest.raises(ValueError):
        Policy(.8, .2)


def test_prompt_is_not_logged(caplog):
    import logging
    with caplog.at_level(logging.INFO, logger='src.predictor.inference'):
        predict_risk('SENSITIVE SYNTHETIC TEXT', MODEL, REVISION, GENERATION_CONFIG, artifact=artifact())
    assert 'SENSITIVE' not in caplog.text
    assert caplog.records[0].predictor_id


def test_mutated_artifact_rejected():
    a = artifact()
    a.generation_config['temperature'] = .9
    with pytest.raises(ValueError):
        predict_risk('Question?', MODEL, REVISION, GENERATION_CONFIG, artifact=a)


def test_median_and_long_input():
    a = Artifact.fit([0, 0, 1], ['a', 'b', 'c'], 'median', {'snapshot': 'b' * 64})
    assert predict_risk('Question?', MODEL, REVISION, GENERATION_CONFIG, artifact=a).estimated_risk == 0
    with pytest.raises(ValueError):
        predict_risk('x' * 100001, MODEL, REVISION, GENERATION_CONFIG, artifact=a)
