"""Run with python -m uvicorn lux_app:app --host 127.0.0.1 --port 8000."""
from pathlib import Path
import logging
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator
from src.predictor.live import LivePredictor

app = FastAPI(title='LUX pre-generation research')
engine = LivePredictor()

class Assessment(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    question: str = Field(min_length=1, max_length=100000)
    context: str = Field(default='', max_length=100000)
    predictor: str = 'prompt_only'

    @field_validator('question')
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError('Question must contain text')
        return value

@app.get('/', response_class=HTMLResponse)
def index():
    return Path(__file__).with_name('lux_ui.html').read_text(encoding='utf-8')

@app.get('/api/health')
def health():
    return {'status':'ok', 'generation_enabled':False}

@app.post('/api/assess-risk')
def assess(data: Assessment):
    try:
        return engine.assess(data.question, data.context, data.predictor)
    except ValueError:
        raise HTTPException(422, 'Invalid input, representation, or incompatible predictor artifact')
    except Exception:
        logging.exception('Assessment failed')
        raise HTTPException(503, 'Assessment unavailable: tokenizer, model, or trained artifact could not be loaded. No baseline fallback was used.')
