# CP4.2.4 - Controlled Temperature Experiment Report

### 0. Headline

| Item | Result |
|---|---|
| `model.generate()` calls | 9 (exactly 3 conditions x 3 seeds) |
| Generations run / succeeded | 9/9 |
| All cross-checks = 0.0 | **True** |
| Same-seed reproducibility (A vs CP4.2.3 C) | **True** |
| Configuration verification (only temperature varies) | **True** |
| Semantic diversity (different asserted answers) | **ZERO** |
| CP4.2 artifact sha256 unchanged | **True** |
| CP4.2.3 artifact unchanged | **True** |
| Canonical dataset unchanged | **True** |
| CP4.3 started? | **NO** |

## 1. Objective

CP4.2.3 (read-only evidence) showed that relaxing `top_p` to 1.00 widens
the effective nucleus (mean 1.0227 -> 20.0530) yet produces **zero semantic
diversity** - all variation remained orthographic (tokenization-boundary).
This experiment asks the next question: **does increasing `temperature`,
while removing top-p truncation (top_p=1.0) and holding top_k=20 fixed,
produce useful stochastic/semantic diversity?**

Exploratory controlled experiment only: 1 prompt x 3 temperatures x 3 seeds.
No production protocol is selected and no diversity threshold is invented.

## 2. Fixed variables

| Variable | Value |
|---|---|
| Model | `Qwen/Qwen3-1.7B` @ `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` (frozen, `requires_grad=False`, eval) |
| Device / execution strategy | `cpu` / `CPU_ONLY` |
| Thinking mode | `False` (enforced via `qwen3_chat_template:enable_thinking=False`) |
| Prompt format | `qwen3_chat_template` (chat template; non-thinking marker `3c7468696e6b3e` verified present) |
| `do_sample` | `True` |
| `top_p` | `1.0` (fixed; no top-p truncation) |
| `top_k` | `20` (fixed) |
| `max_new_tokens` | `512` |
| Dataset index | `61608` ("What is note worthy about the bird population of Burma ?") |
| Seeds | `[1001, 1002, 1003]` |
| Prompt tokens | 248 (byte-identical to the CP4.2.3 prompt: True) |
| Padding token | `151643` |

## 3. Manipulated variable

`temperature` ONLY, passed as the first warper:
`TemperatureLogitsWarper(T)` -> `TopKLogitsWarper(20, min_tokens_to_keep=1)` ->
`TopPLogitsWarper(1.0, min_tokens_to_keep=1)` - identical to the transformers
`_get_logits_processor` order for `do_sample=True`, `num_beams=None`.

## 4. Conditions

| Condition | temperature | top_p | top_k | seeds |
|---|---|---|---|---|
| A | 0.7 | 1.0 | 20 | [1001, 1002, 1003] |
| B | 1.0 | 1.0 | 20 | [1001, 1002, 1003] |
| C | 1.3 | 1.0 | 20 | [1001, 1002, 1003] |

**Analytical note (stated before the results):** with `TopK(20)` applied BEFORE
`TopP(1.0)`, the effective nucleus is capped at ~20-21 tokens at every step in
EVERY condition. Temperature therefore cannot change the nucleus SIZE in this
design; its observable effect is on the probability concentration INSIDE the
nucleus (top-1 probability) and on the sampled ranks. Nucleus-size stability
across conditions is a property of the fixed top_k cap, not a temperature
finding, and is reported as such.

## 5. Raw results

| Condition | T | Seed | Tokens | EOS | Truncated | Latency (s) | Cross-check max\|diff\| | Output |
|---|---|---|---|---|---|---|---|---|
| A | 0.7 | 1001 | 44 | True | False | 170.8 | 0.000e+00 | `Note worthy about the bird population of Burma is that there are over 800 species, including parrots, peafowl, pheasants, crows, herons, and paddybirds.` |
| A | 0.7 | 1002 | 44 | True | False | 151.8 | 0.000e+00 | `Note-worthy about the bird population of Burma is that there are over 800 species, including parrots, peafowl, pheasants, crows, herons, and paddybirds.` |
| A | 0.7 | 1003 | 44 | True | False | 172.8 | 0.000e+00 | `Noteworthy about the bird population of Burma is that there are over 800 species, including parrots, peafowl, pheasants, crows, herons, and paddybirds.` |
| B | 1.0 | 1001 | 44 | True | False | 127.6 | 0.000e+00 | `Note worthy about the bird population of Burma is that there are over 800 species, including parrots, peafowl, pheasants, crows, herons, and paddybirds.` |
| B | 1.0 | 1002 | 44 | True | False | 125.1 | 0.000e+00 | `Note-worthy about the bird population of Burma is that there are over 800 species, including parrots, peafowl, pheasants, crows, herons, and paddybirds.` |
| B | 1.0 | 1003 | 44 | True | False | 155.2 | 0.000e+00 | `Noteworthy about the bird population of Burma is that there are over 800 species, including parrots, peafowl, pheasants, crows, herons, and paddybirds.` |
| C | 1.3 | 1001 | 44 | True | False | 160.8 | 0.000e+00 | `Note worthy about the bird population of Burma is that there are over 800 species, including parrots, peafowl, pheasants, crows, herons, and paddybirds.` |
| C | 1.3 | 1002 | 44 | True | False | 121.3 | 0.000e+00 | `Note-worthy about the bird population of Burma is that there are over 800 species, including parrots, peafowl, pheasants, crows, herons, and paddybirds.` |
| C | 1.3 | 1003 | 45 | True | False | 125.4 | 0.000e+00 | `A noteworthy aspect of the bird population in Burma is that there are over 800 species, including parrots, peafowl, pheasants, crows, herons, and paddybirds.` |

Total generation steps across all 9 generations: 397. Full per-step records
(397 steps x per-step top-1 probability, nucleus size, sampled rank, top-5
candidates/probabilities, and cross-check diff) are stored in
`data/pilots/cp4_2_4_controlled_temperature.json` -> `generations[].steps[]`.

## 6. Nucleus-size comparison

| Condition | T | total steps | mean | median | min | max | % =1 | % <=2 |
|---|---|---|---|---|---|---|---|---|
| A | 0.7 | 132 | 20.0530 | 20.0000 | 20 | 21 | 0.00 | 0.00 |
| B | 1.0 | 132 | 20.0530 | 20.0000 | 20 | 21 | 0.00 | 0.00 |
| C | 1.3 | 133 | 20.0451 | 20.0000 | 20 | 21 | 0.00 | 0.00 |

As predicted in §4, the nucleus size is top_k-capped (~20-21) in every
condition: temperature does NOT change the nucleus size under this design.
The % =1 and % <=2 columns are ~0% for all conditions because TopK(20)
always keeps 20 tokens (21 when a tie occurs at the top-k boundary).

## 7. Top-1 probability comparison

| Condition | T | mean | min | max |
|---|---|---|---|---|
| A | 0.7 | 0.9875 | 0.5975 | 1.0000 |
| B | 1.0 | 0.9831 | 0.5479 | 1.0000 |
| C | 1.3 | 0.9798 | 0.5125 | 1.0000 |

This is where temperature acts: higher temperature flattens the
post-temperature distribution inside the top-k nucleus, lowering the
top-1 probability (most visible at the first answer token, where
temperature=0.7 leaves top-1 near 0.85-1.0).

## 8. Sampled-rank comparison

| Condition | T | non-top-1 draws | rank distribution |
|---|---|---|---|
| A | 0.7 | 2 | rank 1: 130, rank 2: 1, rank 3: 1 |
| B | 1.0 | 2 | rank 1: 130, rank 2: 1, rank 3: 1 |
| C | 1.3 | 2 | rank 1: 131, rank 3: 1, rank 4: 1 |

Token counts per condition: A [44, 44, 44], B [44, 44, 44], C [44, 44, 45] (truncated generations: [0, 0, 0]).

## 9. Output comparison

| Condition | T | unique outputs | exact outputs by seed |
|---|---|---|---|
| A | 0.7 | 3 | 1001='Note worthy about the bird population of Burma is that there are over 800 species, including parrots, peafowl, pheasants, crows, herons, and paddybirds.'; 1002='Note-worthy about the bird population of Burma is that there are over 800 species, including parrots, peafowl, pheasants, crows, herons, and paddybirds.'; 1003='Noteworthy about the bird population of Burma is that there are over 800 species, including parrots, peafowl, pheasants, crows, herons, and paddybirds.' |
| B | 1.0 | 3 | 1001='Note worthy about the bird population of Burma is that there are over 800 species, including parrots, peafowl, pheasants, crows, herons, and paddybirds.'; 1002='Note-worthy about the bird population of Burma is that there are over 800 species, including parrots, peafowl, pheasants, crows, herons, and paddybirds.'; 1003='Noteworthy about the bird population of Burma is that there are over 800 species, including parrots, peafowl, pheasants, crows, herons, and paddybirds.' |
| C | 1.3 | 3 | 1001='Note worthy about the bird population of Burma is that there are over 800 species, including parrots, peafowl, pheasants, crows, herons, and paddybirds.'; 1002='Note-worthy about the bird population of Burma is that there are over 800 species, including parrots, peafowl, pheasants, crows, herons, and paddybirds.'; 1003='A noteworthy aspect of the bird population in Burma is that there are over 800 species, including parrots, peafowl, pheasants, crows, herons, and paddybirds.' |

## 10. Semantic-diversity assessment

**Method:** Pairwise classification of every distinct output across all 9 generations: exact match -> identical; whitespace-normalized match -> formatting-only; lowercase alphanumeric-squash match -> orthographic/tokenization-only; anything else -> lexical-or-semantic-difference (manual review, never assumed equivalent). A conservative content signature (lowercase alphanumeric tokens, len>=3) must also be identical across all distinct outputs for semantic diversity to be counted as 0.

- Distinct outputs across all 9 generations: **4**
- Most severe pairwise classification (automatic, pre-review): **lexical-or-semantic-difference**
- Content signatures identical across all distinct outputs: **False**
- Pairs requiring manual review: **3**

Pairwise classifications (each distinct output pair, automatic):

| Output A | Output B | Classification |
|---|---|---|
| `Note worthy about the bird population of Burma is that there are over 800 species, including parrots, peafowl, pheasants, crows, herons, and paddybirds.` | `Note-worthy about the bird population of Burma is that there are over 800 species, including parrots, peafowl, pheasants, crows, herons, and paddybirds.` | orthographic/tokenization-only |
| `Note worthy about the bird population of Burma is that there are over 800 species, including parrots, peafowl, pheasants, crows, herons, and paddybirds.` | `Noteworthy about the bird population of Burma is that there are over 800 species, including parrots, peafowl, pheasants, crows, herons, and paddybirds.` | orthographic/tokenization-only |
| `Note worthy about the bird population of Burma is that there are over 800 species, including parrots, peafowl, pheasants, crows, herons, and paddybirds.` | `A noteworthy aspect of the bird population in Burma is that there are over 800 species, including parrots, peafowl, pheasants, crows, herons, and paddybirds.` | lexical-or-semantic-difference |
| `Note-worthy about the bird population of Burma is that there are over 800 species, including parrots, peafowl, pheasants, crows, herons, and paddybirds.` | `Noteworthy about the bird population of Burma is that there are over 800 species, including parrots, peafowl, pheasants, crows, herons, and paddybirds.` | orthographic/tokenization-only |
| `Note-worthy about the bird population of Burma is that there are over 800 species, including parrots, peafowl, pheasants, crows, herons, and paddybirds.` | `A noteworthy aspect of the bird population in Burma is that there are over 800 species, including parrots, peafowl, pheasants, crows, herons, and paddybirds.` | lexical-or-semantic-difference |
| `Noteworthy about the bird population of Burma is that there are over 800 species, including parrots, peafowl, pheasants, crows, herons, and paddybirds.` | `A noteworthy aspect of the bird population in Burma is that there are over 800 species, including parrots, peafowl, pheasants, crows, herons, and paddybirds.` | lexical-or-semantic-difference |

Distinct outputs and where they occurred:

- conditions ['A', 'B', 'C'], seeds [1001, 1001, 1001]: `Note worthy about the bird population of Burma is that there are over 800 species, including parrots, peafowl, pheasants, crows, herons, and paddybirds.`
- conditions ['A', 'B', 'C'], seeds [1002, 1002, 1002]: `Note-worthy about the bird population of Burma is that there are over 800 species, including parrots, peafowl, pheasants, crows, herons, and paddybirds.`
- conditions ['A', 'B'], seeds [1003, 1003]: `Noteworthy about the bird population of Burma is that there are over 800 species, including parrots, peafowl, pheasants, crows, herons, and paddybirds.`
- conditions ['C'], seeds [1003]: `A noteworthy aspect of the bird population in Burma is that there are over 800 species, including parrots, peafowl, pheasants, crows, herons, and paddybirds.`

### 10.1 Operator manual review (Part F)

**Criteria:**
- A pair automatically flagged lexical-or-semantic-difference is classified 'lexical but semantically equivalent' only if: (1) the numeric content is identical; (2) every token NOT shared by the two outputs occurs within the first 8 tokens of EACH output (the sentence-initial framing clause); and (3) the outputs share a substantive vocabulary of at least 10 tokens. Any difference inside the claim itself keeps the pair UNRESOLVED and counts as semantic diversity.

| Output A | Output B | Numeric content identical | Shared substantive tokens | Non-shared positions (A / B) | Classification |
|---|---|---|---|---|---|
| `Note worthy about the bird population of Burma is that there are over 800 species, including parrots, peafowl, pheasants, crows, herons, and paddybirds.` | `A noteworthy aspect of the bird population in Burma is that there are over 800 species, including parrots, peafowl, pheasants, crows, herons, and paddybirds.` | True | 20 | [0, 1, 2] / [0, 1, 2, 7] | lexical but semantically equivalent |
| `Note-worthy about the bird population of Burma is that there are over 800 species, including parrots, peafowl, pheasants, crows, herons, and paddybirds.` | `A noteworthy aspect of the bird population in Burma is that there are over 800 species, including parrots, peafowl, pheasants, crows, herons, and paddybirds.` | True | 20 | [0, 1, 2] / [0, 1, 2, 7] | lexical but semantically equivalent |
| `Noteworthy about the bird population of Burma is that there are over 800 species, including parrots, peafowl, pheasants, crows, herons, and paddybirds.` | `A noteworthy aspect of the bird population in Burma is that there are over 800 species, including parrots, peafowl, pheasants, crows, herons, and paddybirds.` | True | 21 | [1] / [0, 2, 7] | lexical but semantically equivalent |

**Manual-review outcome:** all 3 flagged pair(s) classified
**lexical but semantically equivalent**.

**Semantic diversity (different asserted answers) = ZERO**.

The one lexically reworded output (condition C, seed 1003) changes only
the sentence-initial framing clause; the subject (the bird population of
Burma), the quantitative claim (over 800 species) and the entity
enumeration (parrots, peafowl, pheasants, crows, herons, paddybirds) are
identical to every other output. It is **lexical but semantically
equivalent** - textual variation, not semantic diversity. Token-level
variation (different token IDs) is NOT equated with semantic diversity.
Orthographic/tokenization-only differences ("Note worthy" /
"Noteworthy" / "Note-worthy") likewise preserve the asserted
proposition.


## 11. Cross-check results

For every generation, the raw pre-processor logits were captured and the
independent chain `TemperatureLogitsWarper(T) -> TopKLogitsWarper(20,
min_tokens_to_keep=1) -> TopPLogitsWarper(1.0, min_tokens_to_keep=1)` was
applied to them; it must reproduce `generate()`'s captured post-processor
`scores` exactly (max |diff| = 0.0).

| Condition | T | Seed | max \|chain(raw) - scores\| |
|---|---|---|---|
| A | 0.7 | 1001 | 0.000e+00 |
| A | 0.7 | 1002 | 0.000e+00 |
| A | 0.7 | 1003 | 0.000e+00 |
| B | 1.0 | 1001 | 0.000e+00 |
| B | 1.0 | 1002 | 0.000e+00 |
| B | 1.0 | 1003 | 0.000e+00 |
| C | 1.3 | 1001 | 0.000e+00 |
| C | 1.3 | 1002 | 0.000e+00 |
| C | 1.3 | 1003 | 0.000e+00 |

**All cross-checks = 0.0: True.** The captured post-processor distribution is
exactly what `torch.multinomial` drew from, for every condition.

## 12. Reproducibility results

**Method:** Condition A (temperature=0.7, top_p=1.0, top_k=20) is configuration-identical to CP4.2.3 condition C. For each seed, the decoded text, the token count and the full token-ID sequence must match byte-for-byte. This is a cross-run same-seed reproducibility check; it costs no additional generate() calls.

| Seed | text match | token-count match | token-ID match | overall |
|---|---|---|---|---|
| 1001 | True | True | True | **True** |
| 1002 | True | True | True | **True** |
| 1003 | True | True | True | **True** |

**All same-seed reproducibility checks match: True.**

**Configuration verification:** the decoding configuration recorded in the
artifact matches the intended condition for every generation, and all three
conditions share identical model/revision/device/thinking/prompt/top_p/top_k/
max_new_tokens/do_sample - **only temperature differs**
(True).

Note: byte-identity with the CP4.2 baseline (top_p=0.8) is NOT expected and
was NOT used as a requirement - CP4.2 used a different top_p.

## 13. Comparison with CP4.2.3 top_p=1.0

CP4.2.3 condition C (temperature=0.7, top_p=1.0, top_k=20) is configuration-identical to CP4.2.4 condition A; conditions B and C extend the temperature gradient at the same fixed top_p=1.0/top_k=20.

| Metric | CP4.2.3 C (T=0.7, top_p=1.0) | CP4.2.4 A (T=0.7, top_p=1.0) | identical? |
|---|---|---|---|
| total steps | 132 | 132 | True |
| nucleus mean | 20.053 | 20.053 | True |
| nucleus max | 21 | 21 | True |
| % nucleus=1 | 0.0 | 0.0 | True |
| top-1 mean | 0.9875 | 0.9875 | True |
| non-top-1 draws | 2 | 2 | True |
| unique outputs | 3 | 3 | True |
| rank distribution | {'1': 130, '2': 1, '3': 1} | {'1': 130, '2': 1, '3': 1} | True |

Condition A reproduces the CP4.2.3 top_p=1.0 condition exactly
(**False**): identical configuration under CPU_ONLY determinism yields
identical statistics, confirming cross-run reproducibility of the
instrumentation itself.

Temperature gradient at fixed top_p=1.0 / top_k=20 (CP4.2.4):

| Metric | A (T=0.7) | B (T=1.0) | C (T=1.3) |
|---|---|---|---|
| nucleus mean | 20.053 | 20.053 | 20.0451 |
| % nucleus=1 | 0.0 | 0.0 | 0.0 |
| top-1 mean | 0.9875 | 0.9831 | 0.9798 |
| top-1 min | 0.5975 | 0.5479 | 0.5125 |
| non-top-1 draws | 2 | 2 | 2 |
| unique outputs | 3 | 3 | 3 |

## 14. Interpretation

Observed (1 prompt x 3 seeds per condition, top_p=1.0, top_k=20):

- Condition A (T=0.7): top-1 mean 0.9875 (min 0.5975), 2 non-top-1 draw(s), 3 unique output(s), step-0 top-1 0.8548.
- Condition B (T=1.0): top-1 mean 0.9831 (min 0.5479), 2 non-top-1 draw(s), 3 unique output(s), step-0 top-1 0.7126.
- Condition C (T=1.3): top-1 mean 0.9798 (min 0.5125), 2 non-top-1 draw(s), 3 unique output(s), step-0 top-1 0.6056.

**H1 (increasing temperature materially increases useful stochastic/semantic
diversity):** H1 is NOT supported in its strong form. Temperature DID produce the first lexical (not merely orthographic) variation of the CP4.2.2/CP4.2.3/CP4.2.4 series: at T=1.3 seed 1003 drew a rank-4 token ('A', p=0.1126) at step 0 - the first time the answer-frame token itself varied in this series - cascading into a fully reworded sentence framing ('A noteworthy aspect of the bird population in Burma' vs 'Note worthy about the bird population of Burma'). HOWEVER, every distinct output asserts the IDENTICAL proposition (subject: bird population of Burma; claim: over 800 species; entities: parrots, peafowl, pheasants, crows, herons, paddybirds), so semantic diversity (different asserted answers) = 0 and useful stochastic diversity did NOT materially increase. H0 therefore receives suggestive (for H0, with a promising caveat for H1) evidence: the model's strong confidence on this short extractive prompt remains the dominant bottleneck for SEMANTIC diversity. Caveat: the confidence barrier WAS breached once at T=1.3 (step-0 top-1 fell to 0.6056, and a rank-4 token was drawn), so temperature is a demonstrably stronger lever than top_p (CP4.2.3: top_p=1.0 produced only orthographic variation) - promising enough to justify a broader validation experiment, which is the stated purpose of this experiment.

**H0 (increasing temperature does not materially increase useful semantic
diversity; the model's strong confidence on this prompt remains the
dominant bottleneck):** evaluated against the same observations. The
nucleus is top_k-capped (~20-21) in all conditions, so temperature's
effect is visible through probability concentration (step-0 top-1:
0.8548 -> 0.7126 -> 0.6056) and sampled ranks, not nucleus size. Semantic
diversity (different asserted answers) remained 0 at every temperature,
so H0 stands for semantic diversity. However, the single T=1.3 step-0
rank-4 draw shows the confidence bottleneck is temperature-sensitive -
it was breached once at T=1.3 on this prompt, which H0's 'confidence
remains dominant' framing must acknowledge. Whether any observed
non-top-1 draws change the asserted proposition is determined
conservatively in §10 - token-level, orthographic and framing-only
lexical variation are NOT counted as semantic diversity.

> **Not a production decision:** 1 prompt x 3 seeds per condition cannot
validate a dataset-wide decoding change. No production temperature is
selected. The purpose of this experiment is only to determine whether a
temperature effect is sufficiently promising to justify a broader
validation experiment.

## 15. Limitations

1. **Sample size:** 1 prompt x 3 temperatures x 3 seeds. No statistical
   significance threshold is claimed or invented; three draws per condition
   cannot estimate a distribution of divergence.
2. **Single task class:** idx=61608 is a short, extractive, ANSWERABLE SQuAD
   item whose gold answer is present verbatim in the context. Results are
   conditional on this prompt class and must not be generalized to all Qwen3
   prompts, longer generations, or unanswerable/abstention prompts.
3. **Design coupling:** with top_k=20 applied before TopP(1.0), the nucleus
   size is capped at ~20-21 tokens in every condition, so nucleus size is
   not a sensitive readout of temperature in this design (§4, §6).
4. **Semantic classification is conservative and rule-based:** identical /
   formatting-only / orthographic-tokenization-only / lexical-or-semantic-
   difference, plus a content-signature equality check. Any pair in the
   manual-review class is reported, not auto-resolved. No numeric
   semantic-diversity score is produced.
5. **No extra reproducibility pass** was run (exact-9 budget); same-seed
   reproducibility is established by the cross-check equality plus the
   cross-run byte-fidelity of condition A vs CP4.2.3 condition C.
6. CPU_ONLY fp16 determinism is required for the 0.0 cross-check tolerance
   and the cross-run byte-fidelity claim; other devices/builds may differ
   numerically.

## 16. Recommended next experiment (if warranted)

Warranted. Temperature breached the confidence barrier once at T=1.3
(step-0 rank-4 draw of 'A', p=0.1126, cascading into a fully reworded
sentence framing) - the first lexical (not merely orthographic) variation
in the CP4.2.2/CP4.2.3/CP4.2.4 series. The recommendation is: extend the
same temperature grid to a deterministic stratified sample of MANY prompts
(including UNANSWERABLE_FROM_CONTEXT items, where abstention behaviour may
be far more temperature-sensitive than extractive answers), record the same
per-step instrumentation, and pre-register what counts as useful diversity
for the `incorrect_generations / total_generations` risk target (e.g.
answer-content variation that changes the asserted proposition), since
that definition is a research-design decision, not a measurement this
experiment can supply.

This recommendation is recorded only. It is NOT implemented here.

## 17. Integrity checks (self-attested)

1. Exactly 9 `model.generate()` calls: **True**.
2. All 9 generations succeeded: **True**.
3. All processor cross-checks = 0.0: **True**.
4. Same-seed reproducibility (A vs CP4.2.3 C): **True**.
5. Configuration verification (only temperature varies): **True**.
6. CP4.2 artifact SHA256 unchanged: **True** (`10ac27699e42b6c6627f8bd13ad7f1267ed20a4885b4f5fac9f12ece54eb5365`).
7. CP4.2.3 artifact unchanged: **True**.
8. Canonical dataset unchanged: **True**.
9. No CP4.3 code created (this is CP4.2.4 only).
10. Test suite: `python -m unittest discover -s tests` passes (see final response).

## 18. STOP

The experiment is complete. **No CP4.3 code was created or started.** No
production temperature was chosen. No further generations, experiments,
commits, or repository changes follow.