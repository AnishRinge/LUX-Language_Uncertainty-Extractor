# Expanded 500-Prompt SQuAD v2 Cohort

## Scope and preserved artifacts

This cohort expands the existing 31-prompt, 310-generation production cohort
to 500 unique SQuAD v2 training prompts and ten generations per prompt. The
target model remains `Qwen/Qwen3-1.7B`, revision
`70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`, on CPU with thinking disabled,
sampling enabled, temperature 0.7, top-p 0.8, top-k 20, 512 maximum new
tokens, and seeds 1001–1010.

The original `data/production/generations.json`, global hidden states, and all
historical evaluator-version records are read-only inputs. The expanded
generation manifest and copied/reused hidden-state files live under
`data/production/squad_v2_expanded_500/`. Completed original generation
records and CP4.7.0 evaluations are reused without regeneration or relabeling.
The evaluation command writes only CP4.7.0 records to the existing isolated
`data/production/cp4_7_0_gemini_3_5_flash_lite/evaluations/` directory.
CP4.5.0 and other evaluator versions are never combined into the primary
target.

## Selection and measured preflight

Selection is deterministic with seed `20261017`. It preserves each existing
prompt, then uses `random.Random(seed).sample` without replacement from the
remaining unique canonical question/context content pairs. Duplicate content
is detected after Unicode NFKC normalization, case folding, and whitespace
collapsing. Canonical identifiers are retained as prompt IDs; sampled
dataset indices are recorded alongside them.

Measured from the local canonical artifact:

| Item | Count |
| --- | ---: |
| SQuAD v2 training scenarios | 130,319 |
| Unique normalized question/context pairs | 130,219 |
| Duplicate-content rows excluded from the sampling pool | 100 |
| Selected prompts | 500 |
| Existing compatible prompts preserved | 31 |
| New prompts required | 469 |
| Expected total generations | 5,000 |
| Existing compatible generations | 310 |
| Missing Qwen generations | 4,690 |
| Existing selected prompts with compatible hidden states | 31 |
| Selected prompts missing hidden states | 469 |

The fixed evaluator is `gemini-3.5-flash-lite` / `CP4.7.0`. There are 310
matching successful evaluation records for the existing generation cohort,
so the expected additional calls are 4,690, assuming all new generations are
successfully produced and every call resolves once. Retries can increase the
actual number. Historical data include one CP4.7.0 429 rate-limit failure.
No live quota check, cost estimate, or expanded-run runtime estimate was made;
current quota and completion time are unknown.

Version/status counts among evaluation records for the existing generation
IDs at preflight were:

| Evaluator / prompt version | SUCCESS | MANUAL_REVIEW | FAILED |
| --- | ---: | ---: | ---: |
| `gemini-2.5-flash` / `CP4.4.1` | 18 | 0 | 1 |
| `gemini-2.5-flash-lite` / `CP4.6.0` | 0 | 0 | 1 |
| `gemini-3.5-flash-lite` / `CP4.7.0` | 310 | 0 | 0 |
| `gemini-3.8-flash` / `CP4.4.0` | 16 | 0 | 0 |
| `gemini-3.8-flash` / `CP4.5.0` | 82 | 1 | 1 |

The machine-readable selected IDs, indices, seed, and provenance are in
[`selection.json`](../data/production/squad_v2_expanded_500/selection.json).
The measured counts and staged commands are in
[`preflight.json`](../data/production/squad_v2_expanded_500/preflight.json).

## Target construction

The legacy general-purpose target code can divide by the number of resolved
successes, which permits a partial denominator. The versioned CP4.7.0 target
builder is stricter. The expanded cohort explicitly adopts the latter rule:
exactly ten matching `gemini-3.5-flash-lite` / `CP4.7.0` records must be
`SUCCESS` with `RELIABLE` or `UNRELIABLE` labels. The continuous target is
`UNRELIABLE / 10`. `INADEQUATE`, `FAILED`, `MANUAL_REVIEW`, missing, or
otherwise incomplete evaluations make that prompt ineligible; none count as
reliable. A target build rejects evaluation records with mismatched model or
prompt-version provenance. Historical CP4.5.0 results remain separate.

## Predictor inputs and comparison

Predictor features are built from canonical question/context measurements and
the target model's pre-generation hidden states. Generated answers, labels,
evaluation status, claim evaluations, empirical risk, and post-generation
latency are not included. Baseline, prompt-only, hidden-state-only, and fusion
comparisons use a common split by prompt identity. Standardization, PCA, and
model fitting remain training-only. The continuous-risk MAE/RMSE/R² and
paired uncertainty reporting are provided by the existing predictor CLI.

## Five-batch execution

The saved selection is partitioned by its recorded order into five consecutive,
non-overlapping batches of 100 prompts. Batch 1 is selection rows 1–100; every
batch contains ten generation seeds (1001–1010) per prompt. The batch CLI
requires an explicit `--batch-id` from 1 through 5, verifies the fixed
500-prompt selection and frozen generation/evaluator configuration, and
persists a batch ledger under `data/production/squad_v2_expanded_500/batches/`.
Generation is restricted to the selected batch; evaluation accepts only the
completed prefix through its batch ID. Batches after the first are blocked
until all preceding batches are complete.

Batch preflight records the exact prompt IDs and measured existing work. It
checks configured credential-slot names and local SDK availability only; it
does not make a live API request or establish current quota availability. Saved
CP4.7.0 successes are reused, while quota and rate-limit status remains subject
to the provider at execution time.

Run each stage separately from the repository root. Review the batch-specific
preflight before starting the expensive generation stage:

```powershell
& .\.venv\Scripts\Activate.ps1
python .\run_expanded_cohort.py preflight --batch-id 1
```

Generation is resumable and atomically checkpoints each generation. To cap a
single invocation at ten incomplete prompts, pass `--max-new-prompts 10` and
repeat the same Batch 1 command until its ledger reports `completed`. Do not
run a second worker at the same time.

```powershell
python .\run_expanded_cohort.py generate --batch-id 1 --max-new-prompts 10
```

After Batch 1 generation is complete, evaluation reuses existing CP4.7.0
results, checkpoints each result, and is bounded to the first 100 prompts. The
runner's 9,000-second execution budget is retained; if it stops on budget or
quota, inspect the batch ledger and persisted run log before resuming. Re-run
only Batch 1 until its evaluation status is complete. Do not start Batches 2–5
as part of the Batch 1 run.

```powershell
python .\run_eval_cp4_7.py evaluate `
  --batch-id 1 `
  --generation-path .\data\production\squad_v2_expanded_500\generations.json `
  --max-seconds 9000 --spacing 5 `
  --quota-cooldown-seconds 60
```

After evaluation, build the version-matched strict targets, build
pre-generation features, then compare the four baselines:

```powershell
python .\run_eval_cp4_7.py build-targets `
  --generation-path .\data\production\squad_v2_expanded_500\generations.json `
  --target-path .\data\production\squad_v2_expanded_500\targets.json

python .\run_expanded_cohort.py build-features

python .\run_predictor.py compare `
  --features-path .\data\production\squad_v2_expanded_500\predictor_features.json `
  --targets-path .\data\production\squad_v2_expanded_500\targets.json `
  --hidden-states-dir .\data\production\squad_v2_expanded_500\hidden_states `
  --output-dir .\data\production\squad_v2_expanded_500\predictors
```

Do not report predictor results until generation/evaluation completeness,
target eligibility, common split size, and evaluator provenance have been
verified from saved artifacts.
