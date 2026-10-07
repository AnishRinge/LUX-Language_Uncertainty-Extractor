# CP4.2 - Generation Protocol Pilot Report

**Run State:** `COMPLETED`
**Verdict:** **NEEDS_REVISION**
**Started:** `2026-10-05T03:23:44.011716+00:00`  
**Updated:** `2026-10-05T10:04:07.598437+00:00`  
**Target Model:** `Qwen/Qwen3-1.7B`  
**Model Revision:** `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`  
**Execution Strategy:** `CPU_ONLY`  

## 1. Verdict Criteria

The verdict is derived from the checks below. Required failures force **FAIL**;
advisory failures force **NEEDS_REVISION**.

| Check | Severity | Result | Detail |
|---|---|---|---|
| `pilot_run_completed` | required | **PASS** | run_state=COMPLETED |
| `generations_were_attempted` | required | **PASS** | 60 generation(s) recorded |
| `generation_success` | required | **PASS** | 60/60 successful, 0 failed |
| `same_seed_reproducibility` | required | **PASS** | 6/6 matched; 0 technical error(s) |
| `no_forbidden_fields` | required | **PASS** | clean |
| `record_metadata_complete` | required | **PASS** | complete |
| `artifact_persisted` | required | **PASS** | pilot artifact re-read and validated |
| `stochastic_diversity_observed` | advisory | **FAIL** | 4/6 prompt(s) collapsed to < 6 unique output(s) |
| `truncation_within_tolerance` | advisory | **PASS** | truncation rate 0.00% (tolerance 50.00%) |

## 2. Protocol Executed

- Thinking mode: `False` enforced via `qwen3_chat_template:enable_thinking=False`
- Prompt format: `qwen3_chat_template` (chat template, not raw completion)
- `do_sample=True`, `temperature=0.7`, `top_p=0.8`, `top_k=20`
- `max_new_tokens=512`, generations per prompt = `10`
- Seed list: `[1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010]` (the only variable changed between generations of a prompt)
- Selection rule: Deterministic mid-point stratification over the canonical dataset in file order. Positions of each answerability class are collected first, then evenly_spaced_indices(len(positions), k) selects k mid-point indices. No random sampling is used, so selection is reproducible for a given dataset. Class is read from reference.answerability (ANSWERABLE / UNANSWERABLE_FROM_CONTEXT).

## 3. Pilot Selection

| # | dataset_index | sample_id | answerability | article_id |
|---|---|---|---|---|
| 1 | 19919 | `squad_v2_56e7906100c9c71400d772d7` | ANSWERABLE | 65 |
| 2 | 61608 | `squad_v2_5726f19edd62a815002e95e4` | ANSWERABLE | 210 |
| 3 | 108548 | `squad_v2_572f435f947a6a140053c828` | ANSWERABLE | 370 |
| 4 | 31073 | `squad_v2_5ad41e23604f3c001a400625` | UNANSWERABLE_FROM_CONTEXT | 107 |
| 5 | 69929 | `squad_v2_5ad3e4e5604f3c001a3ff58f` | UNANSWERABLE_FROM_CONTEXT | 238 |
| 6 | 108653 | `squad_v2_5a2d64e1f28ef0001a526535` | UNANSWERABLE_FROM_CONTEXT | 370 |

## 4. Generation Summary

- Prompts planned: `6`
- Prompts completed: `6`
- Generations planned: `60`
- Generations attempted: `60`
- Successful: `60`
- Failed: `0`
- Average latency: `393.72s`
- Average token count: `23.7`

## 5. Diversity Metrics

| sample_id | answerability | success | unique | duplicates | unique ratio |
|---|---|---|---|---|---|
| `squad_v2_56e7906100c9c71400d772d7` | ANSWERABLE | 10 | 1 | 9 | 10.00% |
| `squad_v2_5726f19edd62a815002e95e4` | ANSWERABLE | 10 | 2 | 8 | 20.00% |
| `squad_v2_572f435f947a6a140053c828` | ANSWERABLE | 10 | 1 | 9 | 10.00% |
| `squad_v2_5ad41e23604f3c001a400625` | UNANSWERABLE_FROM_CONTEXT | 10 | 1 | 9 | 10.00% |
| `squad_v2_5ad3e4e5604f3c001a3ff58f` | UNANSWERABLE_FROM_CONTEXT | 10 | 1 | 9 | 10.00% |
| `squad_v2_5a2d64e1f28ef0001a526535` | UNANSWERABLE_FROM_CONTEXT | 10 | 2 | 8 | 20.00% |

**Overall:** 60/60 successful, 8 unique outputs, 52 duplicates, unique-output ratio **13.33%**, prompts showing stochastic diversity **2/6**.

## 6. Same-Seed Reproducibility

| sample_id | seed | status | match |
|---|---|---|---|
| `squad_v2_56e7906100c9c71400d772d7` | 1001 | SUCCESS | True |
| `squad_v2_5726f19edd62a815002e95e4` | 1001 | SUCCESS | True |
| `squad_v2_572f435f947a6a140053c828` | 1001 | SUCCESS | True |
| `squad_v2_5ad41e23604f3c001a400625` | 1001 | SUCCESS | True |
| `squad_v2_5ad3e4e5604f3c001a3ff58f` | 1001 | SUCCESS | True |
| `squad_v2_5a2d64e1f28ef0001a526535` | 1001 | SUCCESS | True |

Reproducibility generations are stored separately and are excluded from the
primary generation set.

## 7. Failures

No generation failed.

## 8. Data Leakage Check

Forbidden fields searched in every generation and reproducibility record:
`hallucination_label`, `response_label`, `empirical_risk`, `empirical_hallucination_risk`, `evaluator_output`, `predictor_features`, `hidden_states`, `hidden_state`, `ppl`, `perplexity`.

- Forbidden field violations: `0`
- Artifact re-read validated: `True`
- No hallucination label, empirical risk, evaluator output, predictor feature or
  hidden state is produced by this checkpoint. Seeds are metadata only.

## 9. Recommendation

Required criteria passed but advisory criteria did not. Review before freezing:

- `stochastic_diversity_observed`

CP4.2 produces generation-protocol infrastructure only. No hallucination
evaluation, empirical risk, hidden-state extraction or predictor training is
performed here.
