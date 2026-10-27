"""Versioned prompt provenance for Gemini 3.5 Flash-Lite evaluation."""

from .prompt import (
    build_messages as _build_messages,
    response_schema as _response_schema,
)

EVALUATOR_MODEL = "gemini-3.5-flash-lite"
PROMPT_VERSION = "CP4.7.0"
CHANGE_REASON = (
    "Gemini 2.5 Flash-Lite retirement; selected the model suggested by the "
    "provider's persisted retirement response"
)


def build_messages(question, context, generated_answer):
    """Reuse the existing evidence and labeling instructions unchanged."""
    return _build_messages(question, context, generated_answer)


def response_schema():
    """Reuse the existing structured evaluation schema."""
    return _response_schema()
