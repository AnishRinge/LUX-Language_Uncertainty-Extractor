"""Live inference using the preserved CP4.7 batch-05 trained pipelines."""
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
import hashlib
import json
import numpy as np
from .inference import MODEL, REVISION, Artifact
from .baselines import ALL_FEATURE_NAMES, PromptOnlyPredictor, HiddenStateRow, FusionRow
from .hidden_only import HiddenStateOnlyPredictor
from .fusion import FusionPredictor

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / 'data/production/cp4_7_0_gemini_3_5_flash_lite/predictors_batch_05'


class LivePredictor:
    def __init__(self):
        self.lock = RLock()
        self.tokenizer = self.model = None
        self.predictors = {}

    def assess(self, question, context, predictor='prompt_only'):
        if not isinstance(question, str) or not question.strip() or not isinstance(context, str):
            raise ValueError('A nonempty question and text context are required')
        if max(len(question), len(context)) > 100000:
            raise ValueError('Input exceeds 100000 characters')
        if predictor not in ('prompt_only', 'hidden_only', 'fusion', 'mean', 'median'):
            raise ValueError('Unknown predictor')
        with self.lock:
            return self._assess(question, context, predictor)

    def _assess(self, q, c, name):
        warnings = ['Research estimate; predictive validity and calibration are unverified. A lower estimate does not guarantee better factual correctness.']
        shape = None
        features = None
        if name in ('mean', 'median'):
            path = ROOT / f'docs/engineering/baseline_{name}_v1.json'
            artifact = Artifact.load(path)
            raw = artifact.value
        else:
            from transformers import AutoTokenizer
            if self.tokenizer is None:
                self.tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
            features = dict(question_chars=len(q), context_chars=len(c),
                question_tokens=len(self.tokenizer(q, add_special_tokens=False)['input_ids']),
                context_tokens=len(self.tokenizer(c, add_special_tokens=False)['input_ids']),
                question_ends_with_question_mark=q.rstrip().endswith('?'))
            x = np.array([[float(features[k]) for k in ALL_FEATURE_NAMES]], dtype=np.float32)
            path = ARTIFACTS / name / f'{name}_model.joblib'
            metadata = json.loads((path.parent / 'metadata.json').read_text())
            if set(metadata['train_prompt_ids']) & set(metadata['test_prompt_ids']):
                raise ValueError('Training/test identity overlap')
            if name == 'prompt_only' and metadata['feature_names'] != ALL_FEATURE_NAMES:
                raise ValueError('Incompatible feature schema')
            if name not in self.predictors:
                cls = dict(prompt_only=PromptOnlyPredictor, hidden_only=HiddenStateOnlyPredictor, fusion=FusionPredictor)[name]
                self.predictors[name] = cls.load(path, path.parent / 'metadata.json')
            if name == 'prompt_only':
                raw = float(self.predictors[name].predict(x)[0])
            else:
                import torch
                from transformers import AutoModelForCausalLM
                from run_cp4_2_pilot import render_prompt
                if self.model is None:
                    self.model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, local_files_only=True, torch_dtype=torch.float16).to('cpu').eval()
                    self.model.requires_grad_(False)
                encoded = self.tokenizer(render_prompt(self.tokenizer, c, q), return_tensors='pt')
                if encoded['input_ids'].shape[1] > 4096:
                    raise ValueError('Rendered prompt exceeds interactive limit of 4096 tokens; no truncation performed')
                with torch.inference_mode():
                    states = self.model(**encoded, output_hidden_states=True, use_cache=False).hidden_states
                mask = encoded['attention_mask'][..., None].float()
                pooled = torch.stack([(s.float()*mask).sum(1)/mask.sum(1) for s in states])[:, 0].numpy()
                shape = list(pooled.shape)
                if shape != [29, 2048] or not np.isfinite(pooled).all():
                    raise ValueError('Invalid hidden representation')
                h = pooled[-1]
                row = HiddenStateRow('live', -1, h, 0.) if name == 'hidden_only' else FusionRow('live', -1, features, h, 0.)
                raw = float(self.predictors[name].predict([row])[0])
        if not np.isfinite(raw):
            raise ValueError('Non-finite predictor output')
        score = float(np.clip(raw, 0, 1))
        if score != raw:
            warnings.append('Raw regression output clipped to [0, 1].')
        return dict(risk_score=score, raw_score=raw, predictor_name=name, feature_group=name,
            learned=name not in ('mean','median'), target_model=MODEL, target_revision=REVISION,
            prompt_features=features, hidden_state_status='success' if shape else 'not used', hidden_state_shape=shape,
            pooling='attention-mask token mean; last layer' if shape else None,
            prediction_timestamp=datetime.now(timezone.utc).isoformat(), warnings=warnings,
            provenance={'artifact_sha256':hashlib.sha256(path.read_bytes()).hexdigest(), 'cohort':'CP4.7 batch 05'})
