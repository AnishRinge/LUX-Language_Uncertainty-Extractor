"""Versioned evaluator prompt for CP4.5.0 production evaluation.

The evaluator instructions and response schema remain unchanged from CP4.4.1.
Only the independently hosted evaluator model changed after Gemini 2.5 Flash
access was denied across the configured projects.
"""

from typing import Any, Dict, List

from .prompt import (
    EVALUATOR_SYSTEM_INSTRUCTION,
    build_messages as _build_base_messages,
    response_schema as _base_response_schema,
)

EVALUATOR_MODEL = "gemini-3.8-flash"
PROMPT_VERSION = "CP4.5.0"
CHANGE_REASON = (
    "Gemini 2.5 Flash model access restrictions across all configured projects"
)


def build_messages(question: str, context: str, generated_answer: str) -> List[Dict[str, str]]:
    """Build CP4.5 messages using the unchanged evidence-grounded instructions."""
    return _build_base_messages(question, context, generated_answer)


def response_schema() -> Dict[str, Any]:
    """Return the unchanged structured-label schema for CP4.5."""
    return _base_response_schema()
