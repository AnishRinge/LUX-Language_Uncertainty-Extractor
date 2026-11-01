"""Reproduce the LUX live inference context issue and diagnose root cause."""
import json
import sys
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.predictor.live import LivePredictor
from run_cp4_2_pilot import render_prompt, build_chat_messages, build_user_message

Q = "What is an LLM? Answer in 2 sentences."
CTX = "An LLM is a large language model. It is trained on a large corpus of text."
IRRELEVANT_CTX = "The sky is blue on a clear day."

predictor = LivePredictor()

print("=" * 70)
print("LIVE PREDICTOR ASSESSMENT (reproduction)")
print("=" * 70)
for name in ("prompt_only", "hidden_only", "fusion"):
    for label, ctx in (("NO CONTEXT", ""), ("RELEVANT CONTEXT", CTX), ("IRRELEVANT CONTEXT", IRRELEVANT_CTX)):
        r = predictor.assess(Q, ctx, name)
        print(f"{name:12s} {label:20s} risk={r['risk_score']:.4f} raw={r['raw_score']:.4f} "
              f"features={r['prompt_features']} shape={r['hidden_state_shape']}")
    print()

print("=" * 70)
print("TOKENIZER / RENDER DIAGNOSTIC")
print("=" * 70)
tok = predictor.tokenizer
q_tok = tok(Q, add_special_tokens=False)["input_ids"]
c_tok = tok(CTX, add_special_tokens=False)["input_ids"]
print(f"question tokens (add_special_tokens=False): {len(q_tok)}")
print(f"context tokens  (add_special_tokens=False): {len(c_tok)}")
print(f"question+context separate sum: {len(q_tok) + len(c_tok)}")

# What does the rendered chat prompt look like, and how many tokens?
rendered_no_ctx = render_prompt(tok, "", Q)
rendered_ctx = render_prompt(tok, CTX, Q)
rendered_irr = render_prompt(tok, IRRELEVANT_CTX, Q)
enc_no = tok(rendered_no_ctx, return_tensors="pt")
enc_ctx = tok(rendered_ctx, return_tensors="pt")
enc_irr = tok(rendered_irr, return_tensors="pt")
print(f"\nrendered prompt token length: no-ctx={enc_no['input_ids'].shape[1]} "
      f"relevant={enc_ctx['input_ids'].shape[1]} irrelevant={enc_irr['input_ids'].shape[1]}")
print(f"delta (relevant - no-ctx) = {enc_ctx['input_ids'].shape[1] - enc_no['input_ids'].shape[1]}")
print(f"delta (irrelevant - no-ctx) = {enc_irr['input_ids'].shape[1] - enc_no['input_ids'].shape[1]}")

print("\n--- rendered no-ctx ---")
print(repr(rendered_no_ctx[:400]))
print("\n--- rendered with-ctx (first 600) ---")
print(repr(rendered_ctx[:600]))

print("\n" + "=" * 70)
print("HIDDEN-STATE POOLING COMPARISON (live vs production capture)")
print("=" * 70)
import torch
from run_production import capture_hidden_states

for label, ctx in (("NO CONTEXT", ""), ("RELEVANT CONTEXT", CTX)):
    live_enc = tok(render_prompt(tok, ctx, Q), return_tensors="pt")
    with torch.inference_mode():
        live_states = predictor.model(**live_enc, output_hidden_states=True, use_cache=False).hidden_states
    live_mask = live_enc["attention_mask"][..., None].float()
    live_pooled = torch.stack([(s.float()*live_mask).sum(1)/live_mask.sum(1) for s in live_states])[:, 0].numpy()

    prod_text = render_prompt(tok, ctx, Q)
    prod = capture_hidden_states(predictor.model, tok, prod_text)

    print(f"\n{label}:")
    print(f"  live seq_len={live_enc['input_ids'].shape[1]} prod seq_len={prod['seq_len']}")
    print(f"  live pooled shape={live_pooled.shape} prod mean_pooled shape={np.array(prod['mean_pooled']).shape}")
    print(f"  live last-layer mean  ={live_pooled[-1].mean():.6f} std={live_pooled[-1].std():.6f}")
    print(f"  prod last-layer mean  ={np.array(prod['mean_pooled'])[-1].mean():.6f} std={np.array(prod['mean_pooled'])[-1].std():.6f}")
    print(f"  max abs diff (last layer) = {np.abs(live_pooled[-1] - np.array(prod['mean_pooled'])[-1]).max():.6f}")

print("\n" + "=" * 70)
print("PROMPT-ONLY FEATURE VECTOR COMPARISON")
print("=" * 70)
# What features does the trained prompt_only pipeline expect, and what does
# the live path feed it?
import joblib
ppl = joblib.load(ROOT / "data/production/cp4_7_0_gemini_3_5_flash_lite/predictors_batch_05/prompt_only/prompt_only_model.joblib")
print("pipeline steps:", [s[0] for s in ppl.steps])
scaler = ppl.named_steps["scaler"]
print("scaler mean_:", scaler.mean_)
print("scaler scale_:", scaler.scale_)
for label, ctx in (("NO CONTEXT", ""), ("RELEVANT CONTEXT", CTX), ("IRRELEVANT CONTEXT", IRRELEVANT_CTX)):
        feats = dict(question_chars=len(Q), context_chars=len(ctx),
            question_tokens=len(tok(Q, add_special_tokens=False)["input_ids"]),
            context_tokens=len(tok(ctx, add_special_tokens=False)["input_ids"]),
            question_ends_with_question_mark=Q.rstrip().endswith("?"))
        cols = ["question_chars", "context_chars", "question_tokens", "context_tokens", "question_ends_with_question_mark"]
        x = np.array([[float(feats[k]) for k in cols]], dtype=np.float32)
        pred = float(ppl.predict(x)[0])
        print(f"{label:20s} feats={x[0].tolist()} pred={pred:.4f}")