# CP4.2 — Stochastic Diversity Collapse: Diagnosis

**Document type:** Read-only analysis. No generation was executed, no artifact was modified, no source/config/test file was changed.

**Diagnosis date:** 2026-10-05
**Subject:** CP4.2 verdict `NEEDS_REVISION` — advisory criterion `stochastic_diversity_observed` failed (4/6 prompts collapsed to a single unique output across 10 seeds).
**Artifact analysed:** `data/pilots/cp4_2_generation_pilot.json` (sha256 `10ac27699e42b6c6627f8bd13ad7f1267ed20a4885b4f5fac9f12ece54eb5365`)

---

## 1. Objective

Determine which of the following most plausibly explains the observed diversity collapse:

| | Hypothesis |
|---|---|
| **A** | Selected SQuAD prompts are intrinsically low-entropy / easy |
| **B** | The decoding configuration (`temperature=0.7`, `top_p=0.8`, `top_k=20`) |
| **C** | Seed / sampling implementation defect |
| **D** | Qwen3-1.7B behaviour under this prompt format |
| **E** | Some other implementation or runtime issue |

**Explicitly out of scope:** changing any generation parameter, re-running the pilot, evaluating hallucination, or drawing research conclusions from six prompts.

---

## 2. Evidence Inspected

| Source | What was used |
|---|---|
| `run_cp4_2_pilot.py` (current working copy, 1,105 lines) | `run_generation_pilot`, `_load_model`, `render_prompt`, `evaluate_verdict`, `generation_config`, module constants |
| `tests/test_generation_pilot.py` (current working copy) | Diversity/verdict criteria definitions and their asserted intent |
| `configs/experiment.yaml` (current working copy) | `target_model`, `ground_truth`, `data_leakage` |
| `docs/cp4_2_generation_protocol_pilot_report.md` | Generated verdict criteria table and reported metrics |
| `data/pilots/cp4_2_generation_pilot.json` | All 60 generation records, 6 reproducibility rows, 6 diversity rows, verdict block |
| `data/prompts/squad_v2_canonical_scenarios.json` | The 6 selected scenarios (read-only) |
| `transformers 5.16.1` installed source | `GenerationMixin._get_logits_processor`, `_sample`, `generate`, `_expand_inputs_for_generation` |
| Local HF cache (revision `70d244cc…`) | `config.json`, `generation_config.json` |

**No network research was required** — the local cached configuration files and installed `transformers` source were sufficient.

---

## 3. Generation Implementation Audit

### 3.1 Seeding — correct

```
run_cp4_2_pilot.py:121   torch.manual_seed(seed)          # inside the per-seed loop
run_cp4_2_pilot.py:190   torch.manual_seed(SEEDS[0])      # reproducibility pass
```

`torch.manual_seed` is executed immediately before **every** `generate()` call. There is no path where a generation runs unseeded. Occurrence counts in `run_generation_pilot`: exactly 2 (one per call site family).

### 3.2 `torch.cuda.manual_seed_all()` — irrelevant, and correctly absent

Occurrence count: **0**. Under `CPU_ONLY` (torch `2.14.0+cpu`, `torch.cuda.is_available() == False`, `device_count == 0`) a CUDA seeding call would be a no-op at best. Its absence is **correct behaviour, not a defect**.

### 3.3 Determinism-forcing settings — none present

| Setting | Occurrences |
|---|---|
| `torch.use_deterministic_algorithms` | 0 |
| `CUBLAS_WORKSPACE_CONFIG` | 0 |
| `torch.backends.cudnn.*` | 0 |
| `transformers.set_seed` | 0 |
| `num_beams` | 0 |
| `do_sample=False` | 0 |

No deterministic algorithm mode is enabled anywhere. Greedy decoding is never selected.

### 3.4 Explicit generation kwargs — both call sites identical

```python
model.generate(
    **encoded,
    do_sample=DO_SAMPLE,        # True
    temperature=TEMPERATURE,    # 0.7
    top_p=TOP_P,                # 0.8
    top_k=TOP_K,                # 20
    max_new_tokens=MAX_NEW_TOKENS,  # 512
    pad_token_id=pad_token_id,
)
```

`do_sample=True` **is** passed explicitly at both sites (lines 129–137 and 193–201).

### 3.5 Additional logits processors — none

`repetition_penalty`, `typical_p`, `min_p`, `epsilon_cutoff`, `eta_cutoff`, `bad_words_ids`, `suppress_tokens`, `no_repeat_ngram_size`, `exponential_decay_length_penalty` — **all absent**. None of them constrains sampling here.

### 3.6 Sampling path is genuinely stochastic

From the installed `transformers 5.16.1` `GenerationMixin._sample`:

```python
if do_sample:
    probs = nn.functional.softmax(next_token_scores, dim=-1)
    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
else:
    next_tokens = torch.argmax(next_token_scores, dim=-1)
```

`torch.multinomial` is used; `argmax` is confined to the `else` branch. **Greedy selection cannot occur when `do_sample=True`.**

### 3.7 Warpers actually applied

From `_get_logits_processor`, the `if generation_config.do_sample:` branch appends, in order:

1. `TemperatureLogitsWarper(0.7)`
2. `TopKLogitsWarper(top_k=20, min_tokens_to_keep=1)`
3. `TopPLogitsWarper(top_p=0.8, min_tokens_to_keep=1)`

`min_tokens_to_keep = 1` because `num_beams` is `None`/≤1. No other warper is appended.

### 3.8 `eos_token_id` / `pad_token_id` / `use_cache` / trust_remote_code

- Model `eos_token_id` = `151645` (`<|im_end|>`); `generation_config.json` lists `[151645, 151643]`. A legitimate termination path.
- `pad_token_id` is taken from `tokenizer.pad_token_id` (`151643`), with fallback to `eos_token_id`. Batch size is 1, so padding is not a confound.
- `use_cache` is **not** passed explicitly; `model_kwargs["use_cache"] = generation_config.use_cache` and `config.json` has `use_cache: true`. **Irrelevant to diversity** — caching affects speed, not the sampling distribution.
- `trust_remote_code=True` is passed at load time only (loader), not per-generation. Qwen3 is natively supported in `transformers 5.16.1`, so no remote code executes.

### 3.9 Input tensors are identical across generations — and that is safe

`encoded = tokenizer(prompt_text, return_tensors="pt").to(DEVICE)` is created **once per prompt** (line 107) and reused via `**encoded` for all 10 seeds and the reproducibility pass.

This is correct behaviour (same prompt must produce the same input), and it is **safe**:

- `GenerationMixin._expand_inputs_for_generation` returns `input_ids` **unchanged** when `expand_size == 1` (explicit early return, no clone, no mutation).
- The growth loop uses `input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)`, which **rebinds a local name**; it does not write into the caller's tensor.
- The KV cache is constructed per call via `_prepare_cache_for_generation`, so no state carries between generations.

**Conclusion: no cross-generation state leakage through tensor reuse.**

### 3.10 Could decoding collapse genuinely different token sequences?

Theoretically possible (different token IDs decoding to the same string), but it cannot explain the observed pattern. See §6: for `idx=108653` the token **counts** differ (25 vs 27), which is direct proof that at least one prompt produced genuinely different token sequences, and those differences *did* surface in the decoded text.

### 3.11 Implementation audit verdict

**No implementation defect was found.** Seeding, sampling mode, warper set, tensor reuse, cache handling and termination paths are all correct.

---

## 4. Six-Prompt Output Analysis

All 60 records inspected. `token counts` are per-seed.

### 4.1 `idx=19919` — ANSWERABLE, 117 prompt tokens, article 65
- Question (101 chars): *"What is the acronym for an organization that serves as an accreditation body for engineering schools?"*
- Output, all 10 seeds identical, 19 tokens each:
  `'The acronym for an organization that serves as an accreditation body for engineering schools is ABET.'`
- Unique exact = **1**; normalized = **1**; aggressive-normalized = **1**
- Variation class: **no variation**

### 4.2 `idx=61608` — ANSWERABLE, 248 prompt tokens, article 210
- Question (56 chars): *"What is note worthy about the bird population of Burma ?"*
- Two variants, both 44 tokens:
  - `'Note worthy about the bird population of Burma is that there are over 800 species, …'` (seeds 1001, 1002, 1005, 1006, 1007, 1009) — 6×
  - `'Noteworthy about the bird population of Burma is that there are over 800 species, …'` (seeds 1003, 1004, 1008, 1010) — 4×
- Unique exact = **2**; normalized = **2**; aggressive-normalized = **2**
- Variation class: **orthographic** (tokenisation boundary only; identical semantic content, identical token count)

### 4.3 `idx=108548` — ANSWERABLE, 240 prompt tokens, article 370
- Question (46 chars): *"What is the name of the urban runoff facility?"*
- Output, all 10 identical, 13 tokens each:
  `'The name of the urban runoff facility is SMURFF.'`
- Unique exact = **1**; normalized = **1**; aggressive-normalized = **1**
- Variation class: **no variation**

### 4.4 `idx=31073` — UNANSWERABLE_FROM_CONTEXT, 230 prompt tokens, article 107
- Question (70 chars): *"The glass deposited onto decaying marine matter transformed into what?"*
- Output, all 10 identical, 16 tokens each:
  `'The glass deposited onto decaying marine matter transformed into oil and natural gas.'`
- Unique exact = **1**; normalized = **1**; aggressive-normalized = **1**
- Variation class: **no variation**

### 4.5 `idx=69929` — UNANSWERABLE_FROM_CONTEXT, 208 prompt tokens, article 238
- Question (59 chars): *"Who had the highest GDP per capita during the 20th century?"*
- Output, all 10 identical, 24 tokens each:
  `"Bermuda had one of the world's highest GDP per capita for most of the 20th century."`
- Unique exact = **1**; normalized = **1**; aggressive-normalized = **1**
- Variation class: **no variation**

### 4.6 `idx=108653` — UNANSWERABLE_FROM_CONTEXT, 430 prompt tokens, article 370
- Question (76 chars): *"What is one of the Santa Monica streets seen in It's a Mad, Mad, Mad, World?"*
- Two variants:
  - 27 tokens, markdown italics — seeds **1001, 1003, 1004, 1008, 1010** (5×)
  - 25 tokens, no italics — seeds **1002, 1005, 1006, 1007, 1009** (5×)
- Unique exact = **2**; normalized = **2**; **aggressive-normalized = 1**
- Variation class: **formatting** (asterisk tokens only; identical semantic content)

### 4.7 Aggregate

| Metric | Value |
|---|---|
| Total records | 60 |
| Prompts with byte-identical output across all 10 seeds | **4 / 6** |
| Prompts with any variation | 2 / 6 |
| **Prompts with semantic / content variation** | **0 / 6** |
| Prompts with wording variation | 0 / 6 |
| Prompts with orthographic variation | 1 / 6 |
| Prompts with formatting variation | 1 / 6 |
| Unique outputs (exact, whitespace-normalized) | 8 / 60 |
| Unique outputs under aggressive normalization | **7 / 60** |
| `reached_max_tokens == True` | **0 / 60** |

**The measured "diversity" is entirely cosmetic.** Under aggressive normalization the two "2 unique" prompts collapse to 1 each, so the true semantic diversity across all 60 generations is **zero**.

---

## 5. Prompt-Characteristic Analysis

| idx | Q chars | Ctx chars | Prompt tokens | Gold answer | Gold in context | expected_behavior | phenomenon |
|---|---|---|---|---|---|---|---|
| 19919 | 101 | 223 | 117 | `ABET` | Yes | ANSWER | grounded_contextual_reliability |
| 61608 | 56 | 773 | 248 | `The abundance of birds is notable with over 800 species` | Yes | ANSWER | grounded_contextual_reliability |
| 108548 | 46 | 798 | 240 | `(SMURFF)` | Yes | ANSWER | grounded_contextual_reliability |
| 31073 | 70 | 817 | 230 | *(none)* | n/a | ABSTAIN | grounded_contextual_reliability |
| 69929 | 59 | 725 | 208 | *(none)* | n/a | ABSTAIN | grounded_contextual_reliability |
| 108653 | 76 | 1428 | 430 | *(none)* | n/a | ABSTAIN | grounded_contextual_reliability |

Observations, stated cautiously:

- All six share `phenomenon = grounded_contextual_reliability`, so the pilot contains **exactly one task type**. It is not a diverse prompt sample in any structural sense.
- All three ANSWERABLE items have the gold answer present verbatim in the context, and the questions are short single-fact queries. These are **likely constrained** extractive items.
- Generated answers are frequently near-copies of context spans: `idx=69929` ≈100 % verbatim, `idx=61608` ≈56 %, `idx=108548` ≈49 %. (Heuristic longest-common-substring on alphanumeric-lowercased text; indicative only.)
- The three UNANSWERABLE items have `expected_behavior = ABSTAIN`, yet the model produced confident, context-fluent declarative statements rather than abstentions. **This is flagged for human review and is NOT interpreted here** — evaluation is out of scope for CP4.2.
- Answer lengths are extremely short: **13–44 tokens, mean 23.7**. This is the single most important structural fact for the diversity question (§6).

**Cautious conclusion:** the six prompts are *likely constrained* short-form extractive items. But hypothesis A alone does **not** explain the collapse, because §6 shows the decoding configuration would collapse even a less constrained prompt.

---

## 6. Seed / Sampling Analysis

### 6.1 What the artifact proves

Token counts per seed:

| idx | token counts by seed (1001→1010) | distinct |
|---|---|---|
| 19919 | 19, 19, 19, 19, 19, 19, 19, 19, 19, 19 | 1 |
| 31073 | 16 ×10 | 1 |
| 61608 | 44 ×10 | 1 |
| 69929 | 24 ×10 | 1 |
| 108548 | 13 ×10 | 1 |
| 108653 | **27, 25, 27, 27, 25, 25, 25, 27, 25, 27** | **2** |

**Direct evidence that seeds change the token sequence:**

- `idx=108653` — a clean bimodal split by seed: `{1002, 1005, 1006, 1007, 1009}` produced 25 tokens; `{1001, 1003, 1004, 1008, 1010}` produced 27. Different token counts **cannot** arise from identical token sequences, and the difference surfaced in the decoded text (italic asterisks).
- `idx=61608` — seeds split `{1003, 1004, 1008, 1010}` vs the rest into `Noteworthy` / `Note worthy`, a different first-token segmentation that survives into the text.

Both patterns are seed-correlated, non-uniform, and non-deterministic. This is the signature of genuine multinomial sampling.

### 6.2 What the artifact cannot prove

**Token IDs are not stored in the artifact.** Records contain `generated_answer`, `generated_token_count`, `latency`, `reached_max_tokens` and metadata only — no `output_token_ids`, no per-step log-probabilities, no top-k candidate traces.

Therefore the following **cannot be determined from the existing artifact**:

- Whether the first generated token differed across seeds for the four fully collapsed prompts.
- The exact token sequences behind any generation.
- The per-step probability distribution, entropy, or margin between top-1 and top-2 candidates at any position.
- Whether the four identical-text prompts produced identical token IDs or merely different IDs that decoded identically.

**These were not re-measured, per the no-experiments constraint.** Obtaining them requires a fresh instrumented run and is listed in §12.

### 6.3 Secondary evidence for correct seeding

Same-seed reproducibility matched **6 / 6** exactly. Combined with §6.1, seeding is demonstrably functional in both directions: same seed → same output, different seed → measurable change.

---

## 7. Model / Configuration Analysis

### 7.1 Model defaults on disk vs. what actually ran

`generation_config.json` (revision `70d244cc…`):

```json
{"do_sample": true, "temperature": 0.6, "top_p": 0.95, "top_k": 20,
 "eos_token_id": [151645, 151643], "pad_token_id": 151643}
```

| Setting | Model default | Explicit kwarg | **Winner** |
|---|---|---|---|
| `do_sample` | `true` | `True` | `True` |
| `temperature` | **0.6** | **0.7** | **0.7** (kwarg) |
| `top_p` | **0.95** | **0.8** | **0.8** (kwarg) |
| `top_k` | 20 | 20 | 20 (agree) |

Explicit `generate()` kwargs override `generation_config.json` defaults. The pilot ran at **0.7 / 0.8 / 20**, matching `generation_config()` in the source and the reported verdict table. All 60 records carry `(0.7, 0.8, 20, 512)`.

### 7.2 Can `generation_config` force deterministic behaviour?

**No.** Its `do_sample` is `true`, and every field that could constrain sampling (`repetition_penalty`, `top_k` beyond 20, `min_p`, `typical_p`, `epsilon_cutoff`, `eta_cutoff`, `bad_words_ids`, `num_beams`) is either absent or `null`. `config.json` has no generation-forcing fields.

### 7.3 The mechanism that explains the collapse

This is the analytical core of the diagnosis. Combining the verified configuration with the verified answer lengths:

1. **Answers are extremely short.** 13–44 tokens (mean 23.7). There are only ~13–44 independent sampling decisions available per generation, versus up to 512 if the model had run to the cap.
2. **The task is extractive with a fixed scaffold.** The chat template plus a terse "Be concise" system prompt drives the model into a `The <X> is <Y>.` frame. After the frame is fixed, the remaining freedom is essentially "which context span fills `<Y>`", and the context usually supplies one obvious candidate.
3. **`temperature=0.7` sharpens rather than flattens.** Transformers applies temperature as `logits / T`; dividing by 0.7 multiplies logits by ≈1.43, **increasing** the gap between the top candidate and the rest.
4. **`top_p=0.8` then removes almost everything that remains.** Nucleus filtering keeps the smallest token set whose cumulative probability reaches 0.8. Whenever the top-1 token alone already exceeds 0.8 — typical for a high-confidence extractive continuation from a 1.7B model — the nucleus collapses to a **single token**, and multinomial sampling over a one-token distribution is identical to argmax.
5. **`top_k=20` is effectively inert here.** It is far wider than the nucleus, so `top_p` dominates the effective truncation.
6. **Ten draws is far too few to escape the mode.** Even in the rare case where a second token retains meaningful mass, sampling it 10 times and having all 10 land identically has non-trivial probability.

**Net effect:** `temperature=0.7` + `top_p=0.8` on short extractive answers reduces effective sampling to near-greedy decoding. This is *expected* behaviour of a correctly implemented sampler on an easy, short-answer distribution — not a bug.

### 7.4 Corroborating configuration detail

The model ships `top_p=0.95` / `temperature=0.6` as its own defaults — i.e. the model's authors selected a **much wider** nucleus than the pilot's 0.8. This is circumstantial support that 0.8 is an unusually aggressive setting for this model, though it is not proof.

---

## 8. Diversity-Criterion Assessment

### 8.1 Current definitions (verbatim from source)

```python
# FIX 9: documented, non-arbitrary acceptance criteria.
# With do_sample=True / temperature=0.7 / top_p=0.8 / top_k=20, ten samples of
# the same prompt from a 1.7B model should not collapse to a single string.
# Collapsing to one unique output across ten seeds means sampling is
# effectively deterministic and the protocol is not demonstrating the
# stochastic behaviour the risk target depends on.
MIN_UNIQUE_OUTPUTS_PER_PROMPT = 2

# If most generations hit the 512-token cap the protocol is not eliciting
# terminated answers, which weakens generation-level evaluation downstream.
MAX_ACCEPTABLE_TRUNCATION_RATE = 0.5
```

### 8.2 Assessment

**`MIN_UNIQUE_OUTPUTS_PER_PROMPT = 2` is an ENGINEERING HEURISTIC, not a scientifically justified requirement.** Labelled as such here explicitly:

- It has **no derivation**, **no citation**, and **no empirical calibration**.
- Its justifying comment asserts an expectation ("should not collapse") rather than citing evidence.
- It is **not** tied to the project's actual statistical requirement. The binding constraint is `ground_truth.risk_formula = incorrect_generations / total_generations`, which needs the *fraction* of incorrect generations to be estimable — not a raw unique-output count.
- A threshold of 2 is trivially permissive in one direction (1 unique output fails; 2 passes) yet does not distinguish "1 % semantic diversity" from "90 % semantic diversity". **Two unique outputs that differ only by markdown formatting would pass**, which is precisely what nearly happened here.

**`MAX_ACCEPTABLE_TRUNCATION_RATE = 0.5` is likewise a heuristic.** The comment offers a rationale (termination matters for downstream evaluation), which is reasonable, but 0.5 is an asserted round number with no empirical basis. It passed comfortably (0.00 %), so it did not affect this verdict.

**Neither threshold was changed, and neither should be changed on the basis of this document.** They are recorded here as heuristics so that a future `PASS` is not over-interpreted as scientific validation of the protocol.

### 8.3 Cosmetic defect in the verdict detail string (no effect on the verdict)

```python
else "{0}/{1} prompt(s) collapsed to < {1} unique output(s)".format(
    len(low_diversity), len(diversity_by_prompt) or min_unique_outputs_per_prompt
)
```

The generated report prints `4/6 prompt(s) collapsed to < 6 unique output(s)`. The literal `6` in the threshold slot is `min_unique_outputs_per_prompt = 2` coinciding with `len(diversity_by_prompt) = 6`; the sentence is malformed by construction and would print `< 6` regardless of the configured threshold in this run. **The verdict itself correctly used `2`** (`low_diversity` was computed against `unique_outputs < min_unique_outputs_per_prompt`). Formatting only — no metric or verdict is affected.

---

## 9. Root-Cause Assessment

### The two questions, kept separate

**QUESTION 1 — Is the implementation actually stochastic?**
### **YES.** Confidence: **HIGH**.

Evidence: explicit `do_sample=True`; `torch.multinomial` confirmed as the selection operator with `argmax` confined to the non-sampling branch; `torch.manual_seed` before every call; no determinism-forcing settings; seed-correlated token-count divergence on `idx=108653` and seed-correlated first-token divergence on `idx=61608`; 6/6 same-seed reproducibility. See §3, §6.

**QUESTION 2 — Do these six prompts yield enough useful stochastic variation to estimate empirical hallucination risk?**
### **NO.** Confidence: **HIGH** for these six prompts; the observed variation is entirely cosmetic.

Evidence: 4/6 prompts byte-identical across 10 seeds; the 2 varying prompts differ only by orthography and markdown; zero semantic variation; 0/60 truncation. See §4.

> **Critical distinction:** failure of QUESTION 2 is **not** evidence of failure of QUESTION 1. The sampler worked correctly and the variation it produced was simply too small and too cosmetically different to be useful.

### Hypothesis verdicts

| | Hypothesis | Verdict | Confidence | Reasoning |
|---|---|---|---|---|
| **C** | Seed / sampling implementation defect | **RULED OUT** | **High** | Seeding correct at both call sites; multinomial confirmed; no determinism flags; input-tensor reuse proven safe; observed seed-correlated divergence is positive evidence of correct behaviour. (§3, §6) |
| **E** | Other implementation / runtime issue | **RULED OUT as cause** | **Medium-High** | No runtime error, no OOM, 60/60 SUCCESS, 6/6 reproducibility, artifact atomically persisted and re-read valid, token counts internally consistent. The only runtime anomaly (19,268 s latency) is a host-suspend event orthogonal to sampling. |
| **D** | Qwen3-1.7B behaviour under this prompt format | **CONTRIBUTING** | **Medium** | A 1.7B model on short extractive spans with a "Be concise" system prompt produces high-confidence, low-entropy continuations. Cannot be quantified from the artifact (no logits stored). |
| **A** | Prompts intrinsically low-entropy / easy | **CONTRIBUTING** | **Medium** | All six are one `phenomenon`, all ANSWERABLE golds present verbatim, short single-fact questions, answers often verbatim context spans. But 6 prompts is a small sample and cannot establish "intrinsic" low entropy. |
| **B** | Decoding configuration | **DOMINANT** | **Medium-High** | `top_p=0.8` is the operative constraint; when top-1 mass exceeds 0.8 the nucleus reduces to one token and sampling becomes argmax-equivalent. `temperature=0.7` *sharpens* rather than flattens. The model's own shipped defaults use a far wider nucleus (`top_p=0.95`). §7.3. |

**Most plausible single explanation: B (decoding configuration), amplified by A and D.**

The mechanism is deterministic in effect and fully explained by §7.3: a correctly functioning sampler, restricted to a nucleus that frequently contains one token, applied to answers only 13–44 tokens long, cannot produce meaningful variation — and the pilot's `top_p=0.8` is the tightest single constraint in that chain.

**Confidence that B is the dominant lever: Medium-High, not High.** It cannot be raised to High from existing artifacts, because the decisive quantity — the per-step top-1/top-2 probability mass under this exact configuration — was never recorded. Confirming it requires the instrumented measurement in §12, not a re-run of the same pilot.

---

## 10. What Is Established

1. The sampling implementation is **correct and stochastic** — verified in source and corroborated by seed-correlated token-level divergence.
2. `do_sample=True` is explicitly passed; `torch.multinomial` is the operative selector; no determinism-forcing configuration is active anywhere.
3. The pilot ran at `temperature=0.7`, `top_p=0.8`, `top_k=20`, `max_new_tokens=512` — explicit kwargs overrode the model's `0.6 / 0.95 / 20` defaults, exactly as intended.
4. `thinking_mode=False` was enforced through the real Qwen3 chat template; all 60 records confirm it.
5. Input tensors are identical across generations by design and are **provably not mutated** by `generate()`.
6. Same-seed reproducibility is **6/6 exact**.
7. Across 60 generations there was **zero semantic variation**; the observed variation is orthographic on one prompt and formatting on another.
8. Truncation was **0/60** — answers terminated cleanly on EOS well below the 512 cap.
9. `MIN_UNIQUE_OUTPUTS_PER_PROMPT = 2` and `MAX_ACCEPTABLE_TRUNCATION_RATE = 0.5` are **engineering heuristics**, not scientifically justified thresholds.
10. The `NEEDS_REVISION` verdict was produced by the data-driven logic and was **not** hardcoded.

## 11. What Remains Unknown

1. **Whether the four collapsed prompts produced identical token IDs or different IDs decoding to the same text.** Token IDs are not stored.
2. **Per-step sampling distributions** — entropy, top-1/top-2 margins, and how often the `top_p=0.8` nucleus contained exactly one token. This is the decisive missing measurement for hypothesis B.
3. **Whether the first generated token varied across seeds** for the collapsed prompts.
4. **Whether a wider nucleus alone would restore useful diversity**, versus also requiring longer generations, a different prompt format, or a larger prompt sample.
5. **Generalisation beyond six prompts.** Six items of a single `phenomenon` cannot characterise the dataset's diversity profile.
6. **Whether the advisory threshold of 2 is the right bar at all** for the intended risk estimator — this is a research-design question, not an engineering one.

## 12. Recommendation for the Next Experiment

**No generation parameter is changed by this document.** The following is a *proposal for human approval*, not an action.

**Recommended next step: an instrumented, single-prompt diagnostic run — not another pilot.**

Rationale: the missing evidence in §11 is *measurement*, and it can be obtained far more cheaply than a 60-generation run. One prompt, a small number of seeds, with per-step token IDs and top-k candidate probabilities recorded, would confirm or refute hypothesis B directly.

Specifically, the diagnostic should record, for a single prompt:

- the full output token-ID sequence per seed;
- at each decoding step, the top-5 candidate token IDs and their pre/post-warper probabilities;
- the **effective nucleus size** (how many tokens survive `top_p=0.8` at each step);
- the fraction of steps where the nucleus contained exactly one token.

That single measurement discriminates cleanly between B (nucleus collapses to 1 token) and A/D (nucleus is wide but the model is simply confident).

**Deliberately excluded from this recommendation:** any change to `temperature`, `top_p`, `top_k`, `max_new_tokens`, seeds, model, revision, or thinking mode. Those remain locked pending your review. If the diagnostic confirms hypothesis B, the choice of new decoding parameters — and the scientific question of whether *any* decoding configuration can make short extractive SQuAD answers a viable basis for `incorrect_generations / total_generations` — is a research decision that belongs to you, not to this analysis.

**Also recommended, non-experimental:** record `MIN_UNIQUE_OUTPUTS_PER_PROMPT` and `MAX_ACCEPTABLE_TRUNCATION_RATE` explicitly as heuristics in the source comment, and fix the malformed advisory detail string (§8.3). Both are documentation/cosmetic changes and affect no metric or verdict.

---

## 13. CP4.2.1 Instrumented Diagnostic — Direct Measurement

The CP4.2.1 diagnostic (`run_cp4_2_sampling_diagnostic.py`; on-disk evidence `data/pilots/cp4_2_sampling_diagnostic.json` and `docs/cp4_2_sampling_diagnostic_report.md`) measures the **effective** post-processor distribution that CP4.2's frozen configuration actually produces on exactly one already-selected prompt — `dataset_index 61608`, *"What is note worthy about the bird population of Burma?"* (`ANSWERABLE`). It modifies no CP4.2 artifact, config, source, or dataset record, and does not re-run CP4.2.

**Method (verification instrumentation only — no protocol change):** `output_score=True, output_logits=True, return_dict_in_generate=True` are added purely to capture, per step, the post-processor `scores` (the exact tensor `torch.multinomial` consumed) and the raw pre-processor `logits`. An independently rebuilt chain — `TemperatureLogitsWarper(0.7) → TopKLogitsWarper(20, min_tokens_to_keep=1) → TopPLogitsWarper(0.8, min_tokens_to_keep=1)` — applied to the raw logits reproduced the captured `scores` with **max |diff| = 0.0** for all three seeds. That equality proves `probs = softmax(scores)` is exactly the distribution sampled from, so the nucleus sizes and top-1 probabilities below are measurements of the real run, not a reconstruction.

**Results: 3 generations, 132 steps total.**

| Metric | Value |
|---|---|
| `model.generate()` calls | 3 (one per seed; no other generation) |
| Seeds | 1001, 1002, 1003 (first three of the CP4.2 seed set) |
| Tokens per generation | 44 (all EOS-reached; 0 / 3 truncated) |
| Unique final outputs | 2 (seeds 1001 ≡ 1002; seed 1003 differs by one token) |
| Steps with effective nucleus size = 1 | **129 / 132 (97.73%)** |
| Steps with nucleus size ≤ 2 | 132 / 132 (100.00%) |
| Steps sampling off the top-1 token | 1 / 132 (0.76%) |
| Top-1 probability: min / max / mean | 0.6098 / 1.0000 / 0.9911 |
| Maximum nucleus size (any step, any seed) | 2 |

Per seed: nucleus = 1 at 43 / 44 steps; the single off-top-1 draw belongs to seed 1003 only.

**The single-token divergence mechanism (exactly one multinomial decision in 132):**

- **Step 0, all seeds:** token 9112 ("Note"); nucleus = 1; top-1 probability = **1.0000**. The answer frame is locked deterministically before any stochastic choice is made.
- **Step 1 — the deciding step; nucleus = 2 for all three seeds.** Post-processor distribution:
  - token 27290 (" worthy") — prob **0.6098**, rank 1 (top-1)
  - token 42529 ("worthy") — prob **0.3902**, rank 2
  - seeds 1001, 1002 draw token 27290 → `Note worthy …`
  - seed 1003 draws token 42529 → `Noteworthy …` (the single non-top-1 multinomial draw, 0.76% of all steps)
- **Steps 2–43, all seeds:** nucleus = 1 → continuations are identical once the frame and segmentation are fixed.

The entirety of the inter-seed variation for this prompt — under `do_sample=True, temperature=0.7, top_p=0.8, top_k=20` — reduces to **one multinomial draw at a single step where the nucleus happened to be size 2**, and even that draw only changes a whitespace token boundary ("Note worthy" vs "Noteworthy") with identical downstream text.

**Stochasticity / reproducibility evidence**

- `do_sample=True` is passed explicitly; `torch.multinomial` (not `argmax`) is the operative selector — confirmed in the `transformers 5.16.1` `GenerationMixin._sample` path (see §3.6).
- `torch.manual_seed(seed)` precedes every `generate()` call (§3.1).
- Same-seed reproducibility for the CP4.2 pilot was **6 / 6 exact** (§6.3).
- Diagnostic fidelity: seed-1001 output byte-matches the CP4.2 pilot's `seed=1001, idx=61608` record (44 tokens) — so the output-capture instrumentation does **not** alter sampling. **[PASS]**
- At least one genuine off-top-1 multinomial draw was actually observed (seed 1003, step 1): the 0.3902-mass rank-2 token was drawn — direct proof of real multinomial behaviour under the locked configuration.

**CP4.2 artifact integrity:** sha256 `10ac27699e42b6c667f8bd13ad7f1267ed20a4885b4f5fac9f12ece54eb5365` — **unchanged** (this document was authored from it read-only).

---

## 14. Scope and Non-Generalisation

This remains **one prompt × three seeds** for the diagnostic and **six prompts × ten seeds** for the pilot. It does **not**, by itself, establish a universal production decoding protocol. Limitations of the diagnostic measurement:

- The 129 / 132 single-token-nucleus figure was measured on `idx=61608` over 44 generated tokens. Other prompts and longer continuations can expose wider nuclei; one short extractive answer is not representative of all prompts.
- Three seeds cannot estimate a *distribution* of divergence — only that divergence is possible. The single step-1 draw observed here cannot be generalised into a per-step divergence probability without more seeds / prompts.
- All six pilot prompts share `phenomenon = grounded_contextual_reliability` and are short extractive items (mean 23.7 tokens). Results are therefore conditional on this task class.
- A single observed off-top-1 draw at step 1 (where top-1 held 61%, not >80%) shows the nucleus *can* be size 2 under this configuration, but one observation does not quantify how often the nucleus is size 1 across the dataset.

**Token-level vs. textual vs. semantic variation are distinct, and for this prompt they are distinct in magnitude:**

1. **Token-level stochasticity (yes):** 1 / 132 steps drew the non-top-1 token; 2 unique token sequences across 3 seeds.
2. **Textual variation (minimal):** the two sequences differ by a whitespace token boundary ("Note worthy" vs "Noteworthy") → 2 unique strings.
3. **Semantic variation (zero):** both continuations assert the identical fact ("over 800 species, including parrots, peafowl, pheasants, crows, herons, and paddybirds"). The orthographic difference carries no change in asserted content.

This is the same pattern seen at the pilot level in §4.7 / §9: the variation that *does* occur is cosmetic, not semantic. Confirming hypothesis B dataset-wide and, if so, choosing a new decoding configuration, is a research decision that belongs to the project owners — not an inference this single short-prompt diagnostic can support or refute with high confidence.

---

## Appendix — Scope Compliance

| Constraint | Status |
|---|---|
| `model.generate()` called | **NO** (by this document; §13 figures are from the prior CP4.2.1 run, not re-measured here) |
| New generations created | **NO** |
| Qwen weights loaded | **NO** (tokenizer/config read only; no forward pass by this document) |
| Existing artifact modified | **NO** — sha256 `10ac27699e42b6c667f8bd13ad7f1267ed20a4885b4f5fac9f12ece54eb5365`, mtime `2026-10-05T10:04:07.614547+00:00` unchanged |
| Source files modified | **NO** (by this document; the CP4.2.3 `render_report` regression fix + new test are a separate, separately-documented CP4.2.3 action) |
| Config modified | **NO** |
| Tests modified | **NO** (by this document; CP4.2.3 adds `tests/test_sampling_diagnostic_report.py` separately) |
| Environment modified / packages installed | **NO** |
| Files created | `docs/cp4_2_stochastic_diversity_diagnosis.md` (this document; read-only, extended to add §13–§14) only |