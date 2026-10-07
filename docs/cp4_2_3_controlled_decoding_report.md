# CP4.2.3 - Controlled Decoding Experiment Report

### 0. Headline

| Item | Result |
|---|---|
| `model.generate()` calls | 9 (exactly 3 conditions x 3 seeds) |
| Generations run / succeeded | 9/9 |
| All cross-checks = 0.0 | **True** |
| Condition A seed 1001 == CP4.2 reference (byte-identical) | **True** |
| CP4.2 artifact sha256 unchanged | **True** |
| CP4.3 started? | **NO** |

## 1. Protocol (frozen except `top_p`)

| Field | Value |
|---|---|
| Model | `Qwen/Qwen3-1.7B` @ `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` (frozen) |
| Device / thinking | `CPU_ONLY` / `False` |
| do_sample / temperature / top_k / max_new_tokens | `True` / `0.7` / `20` / `512` |
| dataset_index | `61608` (What is note worthy about the bird population of Burma ?) |
| seeds | `[1001, 1002, 1003]` |
| Varied parameter | `top_p` (only) |
| Conditions | A: 0.80 (CP4.2) / B: 0.95 / C: 1.00 |
| Prompt tokens | 248 |
| Reproducibility pass | NO (exact-9 budget preserved; see §4 note) |

The only per-call difference is the `top_p` kwarg, applied as the third warper:
`TemperatureLogitsWarper(0.7)` -> `TopKLogitsWarper(20, min_tokens_to_keep=1)` ->`TopPLogitsWarper(<condition top_p>, min_tokens_to_keep=1)`, identical to the verified
CP4.2.1 chain except for the final `top_p`.

## 2. Execution outcome

- Model loaded via the CP4.2 `_load_model` path in 15.3s; frozen, eval, no grad.
- Thinking validation: Qwen3 non-thinking marker (`3c7468696e6b3e`) present in the rendered prompt.
- Generations: 9/9 ran; each EOS-reached at 44 tokens (0 truncated).
- CP4.2 artifact sha256 `10ac27699e42b6c6627f8bd13ad7f1267ed20a4885b4f5fac9f12ece54eb5365` — **unchanged**. |

## 3. Processor cross-check (methodology fidelity)

For each generation the raw pre-processor logits were captured, and the independent chain
`TemperatureLogitsWarper(0.7) -> TopKLogitsWarper(20, min_tokens_to_keep=1) -> TopPLogitsWarper(<top_p>, min_tokens_to_keep=1)`
was applied to the raw logits; it must reproduce `generate()`'s captured post-processor `scores` (max |diff| = 0.0).

| Condition | top_p | Seed | max \|chain(raw) - scores\| |
|---|---|---|---|
| A | 0.8 | 1001 | 0.000e+00 |
| A | 0.8 | 1002 | 0.000e+00 |
| A | 0.8 | 1003 | 0.000e+00 |
| B | 0.95 | 1001 | 0.000e+00 |
| B | 0.95 | 1002 | 0.000e+00 |
| B | 0.95 | 1003 | 0.000e+00 |
| C | 1.0 | 1001 | 0.000e+00 |
| C | 1.0 | 1002 | 0.000e+00 |
| C | 1.0 | 1003 | 0.000e+00 |

**All cross-checks = 0.0: True.** The captured post-processor distribution is exactly what `torch.multinomial` drew from.

## 4. Fidelity (Part D)

- **Condition A (top_p=0.80), seed 1001** must be byte-identical to the existing CP4.2 pilot output for idx=61608.
- CP4.2 reference: `Note worthy about the bird population of Burma is that there are over 800 species, including parrots, peafowl, pheasants, crows, herons, and paddybirds.` (44 tokens).
- Experiment output (A, seed 1001): `Note worthy about the bird population of Burma is that there are over 800 species, including parrots, peafowl, pheasants, crows, herons, and paddybirds.` (44 tokens).
- Byte-identical: **True**; token-count match: **True**.
- Same-seed reproducibility within this experiment: the cross-check equality (max|diff|=0.0) and the
  cross-run fidelity above confirm that identical (seed, config) reproduces identical output; no extra
  generation was issued, preserving the exact-9-call budget.

## 5. Per-condition statistics (Part E)

### 5.1 Effective nucleus size (post Temperature+TopK+TopP)

| Condition | top_p | total steps | mean | median | min | max | % =1 | % <=2 | non-top-1 sampled |
|---|---|---|---|---|---|---|---|---|---|
| A | 0.8 | 132 | 1.0227 | 1.0000 | 1 | 2 | 97.73 | 100.00 | 1 |
| B | 0.95 | 132 | 1.0682 | 1.0000 | 1 | 3 | 95.45 | 97.73 | 1 |
| C | 1.0 | 132 | 20.0530 | 20.0000 | 20 | 21 | 0.00 | 0.00 | 2 |

### 5.2 Top-1 probability (post-processor softmax max)

| Condition | top_p | mean | min | max |
|---|---|---|---|---|
| A | 0.8 | 0.9911 | 0.6098 | 1.0000 |
| B | 0.95 | 0.9886 | 0.6098 | 1.0000 |
| C | 1.0 | 0.9875 | 0.5975 | 1.0000 |

### 5.3 Sampled-rank distribution (per condition, 1 = top-1 token drawn)

| Condition | top_p | non-top-1 draws | rank distribution |
|---|---|---|---|
| A | 0.8 | 1 | rank 1: 131, rank 2: 1 |
| B | 0.95 | 1 | rank 1: 131, rank 2: 1 |
| C | 1.0 | 2 | rank 1: 130, rank 2: 1, rank 3: 1 |

### 5.4 Unique final outputs

| Condition | top_p | unique outputs | exact outputs |
|---|---|---|---|
| A | 0.8 | 2 | 1001='Note worthy about the bird population of Burma is that there are over 800 species, including parrots, peafowl, pheasants, crows, herons, and paddybirds.'; 1002='Note worthy about the bird population of Burma is that there are over 800 species, including parrots, peafowl, pheasants, crows, herons, and paddybirds.'; 1003='Noteworthy about the bird population of Burma is that there are over 800 species, including parrots, peafowl, pheasants, crows, herons, and paddybirds.' |
| B | 0.95 | 2 | 1001='Note worthy about the bird population of Burma is that there are over 800 species, including parrots, peafowl, pheasants, crows, herons, and paddybirds.'; 1002='Note worthy about the bird population of Burma is that there are over 800 species, including parrots, peafowl, pheasants, crows, herons, and paddybirds.'; 1003='Noteworthy about the bird population of Burma is that there are over 800 species, including parrots, peafowl, pheasants, crows, herons, and paddybirds.' |
| C | 1.0 | 3 | 1001='Note worthy about the bird population of Burma is that there are over 800 species, including parrots, peafowl, pheasants, crows, herons, and paddybirds.'; 1002='Note-worthy about the bird population of Burma is that there are over 800 species, including parrots, peafowl, pheasants, crows, herons, and paddybirds.'; 1003='Noteworthy about the bird population of Burma is that there are over 800 species, including parrots, peafowl, pheasants, crows, herons, and paddybirds.' |

## 6. Diversity comparison (Part F)

- **Condition A (top_p=0.8):** 2 unique / 132 steps in nucleus=1 (97.73%) ; 1 non-top-1 draws.
- **Condition B (top_p=0.95):** 2 unique / 132 steps in nucleus=1 (95.45%) ; 1 non-top-1 draws.
- **Condition C (top_p=1.0):** 3 unique / 132 steps in nucleus=1 (0.0%) ; 2 non-top-1 draws.

Across conditions the experiment records whether relaxing `top_p` widens the effective
nucleus and produces non-cosmetic token/textual divergence. Per the 1-prompt x 3-seed
limitation, these counts describe one prompt, not a dataset.

### Token vs. textual vs. semantic variation (distinguished)

For each condition the report records **token-level stochasticity** (non-top-1 draws,
unique token sequences), **textual variation** (unique decoded strings), and notes that
textual variation is **not** semantic variation unless the asserted content differs.
Orthographic/markdown differences that preserve the asserted fact are not counted as
semantic diversity without evidence to the contrary (see CP4.2 §9 / §4.7).

## 7. Interpretation & confidence (Part F)

Observed (1 prompt x 3 seeds per condition):

- Condition A (top_p=0.80, the frozen CP4.2 value): nucleus=1 in 97.73% of steps; 1 non-top-1 draw(s); 2 unique output(s).
- Condition B (top_p=0.95): nucleus=1 in 95.45% of steps; 1 non-top-1 draw(s); 2 unique output(s).
- Condition C (top_p=1.00): nucleus=1 in 0.0% of steps; 2 non-top-1 draw(s); 3 unique output(s).

Relaxing `top_p` to 1.0 widens the mean effective nucleus size (A 1.0227 -> C 20.0530), and changes non-top-1 draws (1 -> 2) and unique outputs (2 -> 3).

**Evidence for the hypothesis 'relaxing top_p materially increases useful stochastic diversity':**
suggestive evidence. This is one prompt x three seeds per condition; three draws per condition cannot
establish dataset-level behaviour, so confidence in a general effect is **low**.

**Null interpretation (increasing top_p does not materially resolve the collapse):** consistent
with the CP4.2.1 diagnostic for condition A: even at top_p=0.8 the model is highly confident on this short
extractive answer, so most steps collapse to a single-token nucleus regardless of `top_p`. The dominant
driver is the prompt/answer structure (short, extractive, high-confidence continuation), not the sampler.

> **Not a production decision:** this single short prompt cannot validate a dataset-wide decoding change. Choosing a new production `top_p` (or deciding that no `top_p` can make short extractive SQuAD answers viable for `incorrect_generations/total_generations`) is a research decision for the project owners.

## 8. Integrity checks (self-attested)

1. Exactly 9 `model.generate()` calls: **True**.
2. All 9 generations succeeded: **True**.
3. Condition A seed 1001 fidelity: **True**.
4. All processor cross-checks = 0.0: **True**.
5. CP4.2 artifact SHA256 unchanged: **True** (`10ac27699e42b6c6627f8bd13ad7f1267ed20a4885b4f5fac9f12ece54eb5365`).
6. Canonical dataset unmodified (read-only scenario access).
7. No CP4.3 code created (this is CP4.2.3 only).
8. Test suite: `python -m unittest discover -s tests` passes (see final response).

## 9. STOP

The experiment is complete. **No CP4.3 code was created or started.** No production `top_p` was chosen.
No further generations, experiments, commits, or repository changes follow.