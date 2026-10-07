# CP4.2.1 — Instrumented Sampling Diagnostic: Report

**Status:** Generation and measurement **succeeded**; the markdown report was produced directly from the on-disk JSON artifact.

> **Note on `render_report`:** the diagnostic script's `render_report` (in
> `run_cp4_2_sampling_diagnostic.py`, line 702) contained a one-line format-string
> defect — `_fmt_pct(...)` returns a string such as `"97.73%"` passed to a
> `{0:.1f}` placeholder, raising `ValueError: Unknown format code 'f' for object of type 'str'`.
> The minimal fix was **applied** under the CP4.2.3 Part A authorization: the
> `_fmt_pct(...)` argument was replaced with the raw float fraction
> `100.0 * steps_nucleus_eq_1 / total_steps`, leaving the `{0:.1f}%` placeholder
> (and all scientific aggregates) unchanged. A regression test was added in
> `tests/test_sampling_diagnostic_report.py`, and the full suite passes (see Section 7).
> The script's `render_report` now reproduces an equivalent report from the validated
> `data/pilots/cp4_2_sampling_diagnostic.json`; this file was initially authored
> directly from that artifact so it did not depend on the patched code path.

## 1. Configuration & scope (CP4.2 unchanged)

| Field | Value |
|---|---|
| Model | Qwen/Qwen3-1.7B |
| Revision | 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e (frozen) |
| Device | cpu (CPU_ONLY) |
| Thinking | disabled via Qwen3 chat template (`enable_thinking=False`) |
| do_sample / temperature / top_p / top_k / max_new_tokens | True / 0.7 / 0.8 / 20 / 512 |
| Seeds used | 1001, 1002, 1003 (first three of the CP4.2 seed set) |
| Prompt | dataset_index 61608 — "What is note worthy about the bird population of Burma?" (ANSWERABLE) |
| Prompt selection | via `cp42.select_pilot_scenarios` (same scenario object the CP4.2 pilot used) |
| `model.generate()` calls | 3 (exactly one per seed; no other generation) |

Output capture (adds no sampling change): `output_scores=True, output_logits=True, return_dict_in_generate=True`.

## 2. Execution outcome

- Model loaded via the CP4.2 `_load_model` path in 42.7s; no errors during generation.
- **Thinking validation:** the Qwen3 empty-thinking-block marker (UTF-8 hex `3c7468696e6b3e`, the exact literal CP4.2 checks at `run_cp4_2_pilot.py:957`) was **present** in the rendered prompt for idx=61608 — i.e. thinking is disabled and verified. (The earlier "non-thinking prefix not detected" warning was a false alarm caused by a wrong check literal in the diagnostic; the repaired check uses the correct marker.)
- **Generations:** 3/3 SUCCESS; every run EOS-reached at 44 tokens (no truncation).
- **Fidelity (seed 1001 vs CP4.2):** the diagnostic seed-1001 output byte-matches the CP4.2 pilot's recorded seed-1001 idx=61608 output (44 tokens). This confirms the output-capture flags do **not** alter sampling. **[PASS]**
- **Cross-check (methodology fidelity):** the independently rebuilt chain `TemperatureLogitsWarper(0.7) → TopKLogitsWarper(20, min_tokens_to_keep=1) → TopPLogitsWarper(0.8, min_tokens_to_keep=1)` reproduces `generate()`'s captured post-processor `scores` with max |diff| = **0.0** for all three seeds — the captured `scores` are exactly `processor(raw_logits)`, and `probs = softmax(scores)` is exactly the distribution `torch.multinomial` drew from.
- **CP4.2 artifact:** SHA256 `10ac27699e42b6c6627f8bd13ad7f1267ed20a4885b4f5fac9f12ece54eb5365` — **unchanged**.

**Files produced:** `data/pilots/cp4_2_sampling_diagnostic.json` (valid, complete), `docs/cp4_2_sampling_diagnostic_report.md` (this file).

## 3. Answer-level results

| Seed | Tokens | EOS | Output |
|---|---|---|---|
| 1001 | 44 | yes | "Note worthy about the bird population of Burma is that there are over 800 species, including parrots, peafowl, pheasants, crows, herons, and paddybirds." |
| 1002 | 44 | yes | identical to 1001 |
| 1003 | 44 | yes | "Noteworthy about the bird population of Burma is that there are over 800 species, including parrots, peafowl, pheasants, crows, herons, and paddybirds." |

- 2 unique outputs; seeds 1001 and 1002 are **token-for-token identical**; 1001 vs 1003 diverge starting at step 1.

## 4. Per-step statistics (132 steps total)

| Metric | Value |
|---|---|
| Effective nucleus size = 1 | 129 / 132 (97.73%) |
| Effective nucleus size <= 2 | 132 / 132 (100.00%) |
| Sampled token rank > 1 (off top-1) | 1 / 132 (0.76%) |
| Top-1 probability: min / max / mean | 0.6098 / 1.0000 / 0.9911 |
| Max nucleus size (any step) | 2 |

Per seed: nucleus = 1 at 43/44 steps; off-top-1 samples = 0 (seeds 1001, 1002), 1 (seed 1003).

## 5. First tokens & the single divergence point

- **Step 0 (all seeds):** token 9112 → "Note"; nucleus = 1; top-1 probability = **1.0000**. Fully deterministic — the answer frame is locked at the very first token.
- **Step 1 (the deciding step):** nucleus = 2 for all seeds. Post-processor distribution:
  - token 27290 → " worthy" — probability **0.6098** (top-1, rank 1)
  - token 42529 → "worthy" — probability **0.3902** (rank 2)
  - seeds 1001, 1002 sampled token 27290 (" worthy") → "Note worthy"
  - seed 1003 sampled token 42529 ("worthy") → "Noteworthy" (the single off-top-1 draw, 0.76% of all steps)
- **Steps 2–43:** nucleus = 1 for every seed → continuations are identical once the frame is set.

**The entire observable inter-seed variation (2 unique strings) reduces to one non-top-1 multinomial draw at a single step where the nucleus happened to be size 2.**

## 6. Interpretation (measurement only — no parameter changes recommended yet)

- **Stochastic sampling is functioning.** Multinomial draws are confirmed directly: at step 1 the 0.39-mass rank-2 token was drawn by seed 1003; the cross-check (max |diff| = 0.0) proves the captured distribution is `processor(raw_logits)`.
- **But sampling produces no useful variation for *this* prompt.** 97.73% of steps collapse to a single-token nucleus (top-1 probability = 1.0 at those steps), and the mean top-1 probability is 0.9911. Under `temperature=0.7, top_p=0.8, top_k=20` the top-p nucleus shrinks to one token in essentially every step of this short extractive answer, making multinomial sampling equivalent to greedy except at rare multi-token-nucleus steps.
- The first answer token ("Note", prob 1.0, nucleus 1) pins the "Note.../Noteworthy" frame deterministically; the only freedom is a single token at step 1.
- **Hypothesis assessment (this prompt only; do not generalize from 3 seeds / 1 prompt):**
  - B (decoding configuration, especially `top_p=0.8`): **strongly supported** — this is the direct mechanism observed (nucleus → 1 token).
  - C (seed/sampling defect): **ruled out** (fidelity match + cross-check = 0.0 + 6/6 CP4.2 reproducibility).
  - E (runtime issue): **ruled out** (clean run, no errors; the `torch_dtype` deprecation is a harmless warning).
  - A (intrinsically low-entropy prompts) and D (Qwen3-1.7B behaviour): **not separable** here — three seeds of one prompt cannot isolate prompt difficulty from model behaviour; the measured concentration is consistent with both.

This measurement is **not** sufficient to confirm or refute the decoding-collapse hypothesis at the dataset level (3 seeds, 1 prompt). It confirms the *mechanism* on one prompt.

## 7. Format-string defect — fixed & verified

`render_report` in `run_cp4_2_sampling_diagnostic.py` (line 700–702):

```python
"...high ({0:.1f}% here)...".format(_fmt_pct(steps_nucleus_eq_1 / total_steps))
```

`_fmt_pct(...)` returns a string such as `"97.73%"`, which cannot satisfy `{0:.1f}` (expects a float) → `ValueError: Unknown format code 'f' for object of type 'str'`.

**Minimal fix (applied under CP4.2.3 Part A; scientific calculations untouched):** replaced `_fmt_pct(steps_nucleus_eq_1 / total_steps)` — a string — with the raw float fraction `100.0 * steps_nucleus_eq_1 / total_steps`, leaving the `{0:.1f}%` placeholder to format it as `97.7`.

**Verification:** `tests/test_sampling_diagnostic_report.py` loads
`data/pilots/cp4_2_sampling_diagnostic.json` and asserts `render_report(...)` returns
a `str` and contains `(97.7% here)`. `python -m unittest discover -s tests` passes,
including this regression test. The fix was applied **without rerunning `model.generate()`**;
CP4.2.1's diagnostic generation was not re-executed, and no CP4.2 artifact was modified.

## 8. Test suite

`python -m unittest discover -s tests` → **111 tests, 0 failures** (110 prior + 1 regression test for `render_report`). No CP4.2 test was modified. A regression test for the `render_report` format defect was added in `tests/test_sampling_diagnostic_report.py` (validates `render_report` runs on the artifact and emits the numeric `(97.7% here)`); the processor-chain-shape and non-thinking-marker repairs are validated by (a) the successful diagnostic run above and (b) the existing pure-helper unit tests in `tests/test_sampling_diagnostic.py`.
