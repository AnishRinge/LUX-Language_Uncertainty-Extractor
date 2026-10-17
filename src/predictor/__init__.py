"""LUX predictor infrastructure.

Public API:
  - baselines: PromptOnlyPredictor, load_predictor_targets, compute_metrics,
    prompt_only_train_test_split, run_prompt_only_baseline,
    train_test_split_by_prompt, load_hidden_states, extract_hidden_representation
  - hidden_only: HiddenStateOnlyPredictor, run_hidden_only_baseline
  - fusion: FusionPredictor, run_fusion_baseline
"""

from .baselines import (
    PromptOnlyPredictor,
    load_predictor_targets,
    compute_metrics,
    prompt_only_train_test_split,
    run_prompt_only_baseline,
    train_test_split_by_prompt,
    load_hidden_states,
    extract_hidden_representation,
    HiddenStateRow,
    FusionRow,
    AlignmentError,
    MIN_TRAIN_PROMPTS,
    MIN_TEST_PROMPTS,
)
from .hidden_only import (
    HiddenStateOnlyPredictor,
    run_hidden_only_baseline,
)
from .fusion import (
    FusionPredictor,
    run_fusion_baseline,
)

__all__ = [
    "PromptOnlyPredictor",
    "HiddenStateOnlyPredictor",
    "FusionPredictor",
    "load_predictor_targets",
    "compute_metrics",
    "prompt_only_train_test_split",
    "train_test_split_by_prompt",
    "load_hidden_states",
    "extract_hidden_representation",
    "HiddenStateRow",
    "FusionRow",
    "AlignmentError",
    "MIN_TRAIN_PROMPTS",
    "MIN_TEST_PROMPTS",
    "run_prompt_only_baseline",
    "run_hidden_only_baseline",
    "run_fusion_baseline",
]
