import pytest
from fastapi.testclient import TestClient
from lux_app import app
from src.predictor.live import LivePredictor

client = TestClient(app)

@pytest.mark.parametrize('payload', [{}, {'question':' '}, {'question':12}, {'question':'Q','generated_answer':'future'}])
def test_invalid(payload):
    assert client.post('/api/assess-risk', json=payload).status_code == 422

def test_real_baseline_without_generation():
    engine = LivePredictor()
    first = engine.assess('Who?', '', 'mean')
    second = engine.assess('A different question?', 'Longer context', 'mean')
    assert first['risk_score'] == second['risk_score']
    assert not first['learned'] and engine.model is None
    assert 0 <= first['risk_score'] <= 1

def test_api_error_is_sanitized(monkeypatch):
    import lux_app
    def fail(*args):
        raise RuntimeError('secret/path')
    monkeypatch.setattr(lux_app.engine, 'assess', fail)
    response = client.post('/api/assess-risk', json={'question':'Q?'})
    assert response.status_code == 503
    assert 'secret' not in response.text

def test_missing_artifacts_no_fallback(monkeypatch, tmp_path):
    import lux_app
    import src.predictor.live as live
    engine = LivePredictor()
    class Tokenizer:
        def __call__(self, *args, **kwargs):
            return {'input_ids':[1]}
    engine.tokenizer = Tokenizer()
    monkeypatch.setattr(live, 'ARTIFACTS', tmp_path)
    monkeypatch.setattr(lux_app, 'engine', engine)
    response = client.post('/api/assess-risk', json={'question':'Q?', 'predictor':'fusion'})
    assert response.status_code == 503
    assert 'risk_score' not in response.json()
    assert 'No baseline fallback' in response.json()['detail']
    assert not engine.predictors

def test_unknown_predictor_no_fallback():
    response = client.post('/api/assess-risk', json={'question':'Q?', 'predictor':'missing'})
    assert response.status_code == 422
    assert 'risk_score' not in response.json()

def test_actual_prompt_artifact_and_edits():
    engine = LivePredictor()
    a = engine.assess('Who?', 'Paris.')
    b = engine.assess('What is the capital of France?', 'Paris is the capital of France.')
    assert a['learned'] and b['learned']
    assert a['prompt_features'] != b['prompt_features']
    assert a['risk_score'] != b['risk_score']
    assert engine.model is None

@pytest.mark.parametrize('value', [float('nan'), float('inf')])
def test_nonfinite_rejected(value):
    engine = LivePredictor()
    class Tokenizer:
        def __call__(self, *args, **kwargs):
            return {'input_ids':[1]}
    class Predictor:
        def predict(self, x):
            return [value]
    engine.tokenizer = Tokenizer()
    engine.predictors['prompt_only'] = Predictor()
    with pytest.raises(ValueError, match='Non-finite'):
        engine.assess('Q?', '')

def test_hidden_forward_only():
    import torch
    from types import SimpleNamespace
    class Tokenizer:
        def __call__(self, text, **kwargs):
            if kwargs.get('return_tensors'):
                return {'input_ids':torch.tensor([[1,2,0]]), 'attention_mask':torch.tensor([[1,1,0]])}
            return {'input_ids':[1,2]}
        def apply_chat_template(self, *args, **kwargs):
            return 'rendered input'
    class Model:
        def __call__(self, **kwargs):
            assert kwargs['output_hidden_states'] and not kwargs['use_cache']
            state = torch.ones(1,3,2048)
            state[:,2] = 999
            return SimpleNamespace(hidden_states=tuple(state for _ in range(29)))
        def generate(self, **kwargs):
            pytest.fail('Generation forbidden')
    class Predictor:
        def predict(self, rows):
            assert rows[0].representation.shape == (2048,)
            assert (rows[0].representation == 1).all()
            return [0.3]
    engine = LivePredictor()
    engine.model, engine.tokenizer = Model(), Tokenizer()
    engine.predictors['hidden_only'] = Predictor()
    result = engine.assess('Q?', 'Context', 'hidden_only')
    assert result['hidden_state_shape'] == [29,2048]
    assert result['risk_score'] == .3
