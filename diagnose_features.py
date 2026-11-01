"""Diagnose which features drive the context effect in the trained predictors."""
import json
import sys
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import joblib
from sklearn.decomposition import PCA

BATCH = ROOT / "data/production/cp4_7_0_gemini_3_5_flash_lite/predictors_batch_05"

# --- 1. Prompt-only Ridge coefficients ---
ppl = joblib.load(BATCH / "prompt_only" / "prompt_only_model.joblib")
ridge = ppl.named_steps["ridge"]
scaler = ppl.named_steps["scaler"]
cols = ["question_chars", "context_chars", "question_tokens", "context_tokens", "question_ends_with_question_mark"]
print("PROMPT-ONLY RIDGE")
print("  coef (raw units):", np.round(ridge.coef_, 4))
print("  coef * scale (per-SD contribution):", np.round(ridge.coef_ * scaler.scale_, 4))
print("  intercept:", round(ridge.intercept_, 4))
print("  => context_chars coefficient (per char):", ridge.coef_[1])
print()

# --- 2. Load training data to check real correlation ---
feat_path = ROOT / "data/production/predictor_features.json"
tgt_path = ROOT / "data/production/targets.json"
feat_data = json.loads(feat_path.read_text())
tgt_data = json.loads(tgt_path.read_text())
targets_by_idx = {t["dataset_index"]: t for t in tgt_data["targets"]}

rows = []
for e in feat_data["predictor_features"]:
    t = targets_by_idx.get(e["dataset_index"])
    if t and t.get("empirical_hallucination_risk") is not None:
        m = e["prompt_metrics"]
        rows.append((e["prompt_id"], m.get("question_chars",0), m.get("context_chars",0),
                     m.get("question_tokens",0), m.get("context_tokens",0),
                     m.get("question_ends_with_question_mark",0), t["empirical_hallucination_risk"]))
rows.sort(key=lambda r: r[2])
print(f"TRAINING DATA: {len(rows)} prompts")
print(f"  context_chars: min={rows[0][2]} median={np.median([r[2] for r in rows]):.0f} max={rows[-1][2]}")
print(f"  context_chars quartiles: {np.percentile([r[2] for r in rows], [0,25,50,75,100]).round(0)}")
cc = np.array([r[2] for r in rows], dtype=float)
yy = np.array([r[6] for r in rows], dtype=float)
print(f"  corr(context_chars, risk) = {np.corrcoef(cc, yy)[0,1]:.4f}")
print(f"  corr(context_tokens, risk) = {np.corrcoef(np.array([r[4] for r in rows],float), yy)[0,1]:.4f}")
print(f"  corr(question_chars, risk) = {np.corrcoef(np.array([r[1] for r in rows],float), yy)[0,1]:.4f}")

# Bin by context length and show mean risk
bins = [0, 100, 200, 400, 800, 10000]
print("\n  Mean risk by context_chars bin:")
for i in range(len(bins)-1):
    sel = [r[6] for r in rows if bins[i] <= r[2] < bins[i+1]]
    if sel:
        print(f"    [{bins[i]},{bins[i+1]}): n={len(sel):4d} mean_risk={np.mean(sel):.4f}")

# --- 3. Hidden-only: does hidden state correlate with context length? ---
print("\nHIDDEN-ONLY PCA loadings vs context length")
hpl = joblib.load(BATCH / "hidden_only" / "hidden_only_model.joblib")
print("  hidden pipeline steps:", [s[0] for s in hpl.steps])
# Check whether the first PCA component is dominated by sequence-length effects
# by correlating PCA components with context length on training data
import glob
hs_dir = ROOT / "data/production/hidden_states"
hs_files = sorted(hs_dir.glob("hs_*.json"))
print(f"  hidden state files on disk: {len(hs_files)}")

# --- 4. Demonstrate the extrapolation problem ---
print("\nEXTRAPOLATION CHECK (live inputs vs training distribution)")
print(f"  Training context_chars mean={scaler.mean_[1]:.1f} sd={scaler.scale_[1]:.1f}")
for label, ctx_chars in (("empty (0)", 0), ("short (74)", 74), ("typical (~744)", 744)):
    z = (ctx_chars - scaler.mean_[1]) / scaler.scale_[1]
    print(f"  context_chars={ctx_chars:4d} -> z-score={z:+.3f}")