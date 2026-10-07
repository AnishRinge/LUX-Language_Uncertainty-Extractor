# CP4.5-A — Live Gemini Evaluator Smoke Test Report

## Status
Live smoke test executed (6 evaluations).

## Run metadata
- evaluator model: `gemini-3.8-flash`
- source artifact: `data\pilots\cp4_2_5_representative_validation.json`
- selected dataset_index values: [19919, 61608, 108548, 31073, 69929, 108653]
- generated at (UTC): 2026-10-06T16:21:25.972376+00:00

## Request-input constraint verification
The evaluator was built so `build_messages` is called with ONLY `question`,
`context`, and `generated_answer`. No Qwen3 model identity, seed, decoding
config, logits, hidden states, or predictor features are sent to the Gemini API.

## Summary
- successful evaluations: 0
- manual-review evaluations: 0
- failed evaluations: 6
- total retries: 18
- generation-label distribution: {'RELIABLE': 0, 'UNRELIABLE': 0, 'INADEQUATE': 0, None: 6}

## Records
| dataset_index | status | label | claims | review | retries |
|---|---|---|---|---|---|
| 19919 | FAILED | None | 0 | False | 3 |
| 61608 | FAILED | None | 0 | False | 3 |
| 108548 | FAILED | None | 0 | False | 3 |
| 31073 | FAILED | None | 0 | False | 3 |
| 69929 | FAILED | None | 0 | False | 3 |
| 108653 | FAILED | None | 0 | False | 3 |

## Artifact
`data\pilots\cp4_5_a_gemini_smoke_test.json`
