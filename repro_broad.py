"""Broader reproduction: multiple questions x context lengths x predictors."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.predictor.live import LivePredictor

predictor = LivePredictor()

QUESTIONS = [
    "What is an LLM? Answer in 2 sentences.",
    "What is the capital of France?",
    "Who wrote Romeo and Juliet?",
    "What is photosynthesis?",
]
SHORT_CTX = "An LLM is a large language model. It is trained on a large corpus of text."
LONG_CTX = ("The mitochondrion is a double-membrane-bound organelle found in all "
            "eukaryotic organisms. It is often described as the powerhouse of the cell "
            "because it generates most of the cell's supply of adenosine triphosphate, "
            "which is used as a source of chemical energy. Mitochondria also have their "
            "own genetic material, separate from the nuclear DNA. This supports the "
            "endosymbiotic theory, which proposes that mitochondria were once free-living "
            "prokaryotic organisms that were engulfed by an ancestral eukaryotic cell. "
            "They are involved in many cellular processes including calcium signaling, "
            "cellular differentiation, cell death, and the control of the cell cycle.")
IRRELEVANT = "The sky is blue on a clear day. Water is wet."

print(f"{'predictor':11s} | {'question':30s} | {'context':12s} | {'risk':>7s}")
print("-" * 75)
for qi, q in enumerate(QUESTIONS):
    for label, ctx in (("none", ""), ("short", SHORT_CTX), ("long", LONG_CTX), ("irrel", IRRELEVANT)):
        for name in ("prompt_only", "hidden_only", "fusion"):
            r = predictor.assess(q, ctx, name)
            print(f"{name:11s} | {q[:28]:30s} | {label:12s} | {r['risk_score']*100:6.1f}%")
    print()