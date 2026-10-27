"""Versioned prompt provenance for Gemini 2.5 Flash-Lite evaluation."""

from .prompt import (
    build_messages as _build_messages,
    response_schema as _response_schema,
)

EVALUATOR_MODEL = "gemini-2.5-flash-lite"
PROMPT_VERSION = "CP4.6.0"
CHANGE_REASON = (
    "Gemini 2.5 Flash retirement; Flash-Lite selected as a separately "
    "versioned lower-cost evaluator candidate"
)


def build_messages(question, context, generated_answer):
    """Reuse the existing evidence and labeling instructions unchanged."""
    return _build_messages(question, context, generated_answer)


def response_schema():
    """Reuse the existing structured evaluation schema."""
    return _response_schema()
