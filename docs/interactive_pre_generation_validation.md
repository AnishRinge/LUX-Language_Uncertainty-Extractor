# Interactive pre-generation validation ? partial audit

Environment: Windows PowerShell, activated existing .venv, Python 3.11.5. Initial activation was blocked by execution policy; process-scoped Bypass resolved it. No alternate Python was intentionally used after activation.

## Current inference sequence
src/predictor/inference.py validates question/context, frozen model revision, generation configuration and baseline artifact integrity, then returns the artifact constant. No generation, evaluator, tokenizer or Qwen forward pass occurs. This is pre-generation inference without prompt-specific discrimination. Hidden states are not used. No future-answer features enter this route.

run_production.capture_hidden_states performs a forward pass before generation, saving last-token and token-mean representations. Its single unpadded sequence implementation averages all tokens; this is not a verified padded-batch implementation. The historical feasibility report records 29 entries of width 2048; this session has not reproduced actual Qwen outputs.

## Saved OOF metrics (not rerun experiments)
Source: docs/engineering/oof_analysis_v1.json. These are historical saved OOF metrics, not a new held-out evaluation. Model-selection effects and artifact compatibility remain to be audited.

| Predictor / features | MAE | RMSE | R? |
|---|---:|---:|---:|
| dummy_mean|prompt | 0.317064 | 0.386904 | -0.000659 |
| dummy_median|prompt | 0.208642 | 0.439463 | -0.290993 |
| dummy_mean|hidden | 0.317064 | 0.386904 | -0.000659 |
| dummy_median|hidden | 0.208642 | 0.439463 | -0.290993 |
| dummy_mean|fusion | 0.317064 | 0.386904 | -0.000659 |
| dummy_median|fusion | 0.208642 | 0.439463 | -0.290993 |
| ridge|prompt | 0.317543 | 0.388087 | -0.006788 |
| lasso|prompt | 0.318938 | 0.388959 | -0.011315 |
| elasticnet|prompt | 0.318930 | 0.388945 | -0.011245 |
| svr|prompt | 0.214182 | 0.434953 | -0.264634 |
| knn|prompt | 0.306470 | 0.431370 | -0.243884 |
| dt|prompt | 0.333210 | 0.454427 | -0.380409 |
| rf|prompt | 0.324719 | 0.406537 | -0.104793 |
| et|prompt | 0.319657 | 0.395347 | -0.044810 |
| gbr|prompt | 0.324823 | 0.401877 | -0.079608 |
| hgb|prompt | 0.321112 | 0.398047 | -0.059127 |
| mlp|prompt | 0.314222 | 0.400365 | -0.071498 |
| ridge|hidden | 0.394854 | 0.497664 | -0.655590 |
| lasso|hidden | 0.317064 | 0.386904 | -0.000659 |
| elasticnet|hidden | 0.317521 | 0.387765 | -0.005119 |
| svr|hidden | 0.223997 | 0.422184 | -0.191469 |
| knn|hidden | 0.310406 | 0.424087 | -0.202233 |
| dt|hidden | 0.311680 | 0.474396 | -0.504392 |
| rf|hidden | 0.303474 | 0.383895 | 0.014843 |
| et|hidden | 0.310578 | 0.384885 | 0.009756 |
| gbr|hidden | 0.300624 | 0.392408 | -0.029333 |
| hgb|hidden | 0.305694 | 0.396638 | -0.051645 |
| mlp|hidden | 0.717316 | 0.897665 | -4.386518 |
| ridge|fusion | 0.396193 | 0.498794 | -0.663114 |
| lasso|fusion | 0.317064 | 0.386904 | -0.000659 |
| elasticnet|fusion | 0.317521 | 0.387765 | -0.005119 |
| svr|fusion | 0.224017 | 0.422202 | -0.191574 |
| knn|fusion | 0.305968 | 0.418881 | -0.172900 |
| dt|fusion | 0.296684 | 0.464977 | -0.445247 |
| rf|fusion | 0.304071 | 0.384873 | 0.009818 |
| et|fusion | 0.309409 | 0.383472 | 0.017014 |
| gbr|fusion | 0.318777 | 0.410049 | -0.123960 |
| hgb|fusion | 0.310777 | 0.394537 | -0.040530 |
| mlp|fusion | 0.865704 | 1.122945 | -7.429400 |

## Verdict and remaining work
The current live baseline implementation is honest about its constant output. It does not implement the requested prompt/hidden/fused interactive assessment. Predictive validity is not established by this audit. Full artifact provenance checks, controlled prompt diagnostics, actual Qwen integration, API/frontend implementation and their tests remain incomplete. No research data or existing artifacts were overwritten.

## Implemented live interface (2026-10-28)

The earlier partial-audit verdict above describes the pre-change state.
`src/predictor/live.py` now loads preserved CP4.7 batch-05 prompt-only,
hidden-only and fused trained artifacts. No retraining or dataset mutation was
needed. Training/test prompt-ID disjointness is checked on load; prompt schema
is checked for prompt-only. Existing predictor loaders preserve fitted scaling,
PCA and Ridge parameters. Artifact SHA256 is returned. Full cryptographic
binding of old training provenance to the Qwen checkpoint remains absent from
legacy metadata; compatibility relies on the existing frozen production protocol.

Sequence: validate input -> frozen tokenizer -> five input-only features ->
(for hidden/fused) frozen Qwen prompt rendering, CPU float16 forward pass ->
attention-masked token means -> reject non-finite or non-29x2048 representation
-> last layer -> saved preprocessing and trained regressor -> finite check ->
explicit [0,1] clipping -> response. No generate call or evaluator is used.
Baseline selection is explicit and bypasses feature/model extraction.
The API forbids extra fields, including generated answers. Errors are sanitized
and never trigger fallback. One locked engine caches tokenizer/model/predictors.

Actual integration: question `What is the capital of France?`, context `Paris is
the capital of France.` yielded features 30/31 characters, 7/7 tokens and true
question mark. Real prompt-only risk: 0.21740319587972134. Real fused risk:
0.47428810596466064, hidden shape [29,2048]. Qwen weights loaded from local cache
in approximately 24 seconds; full forward and saved fused inference completed.
These are diagnostic scores, not correctness labels or evidence of accuracy.

Application launched with `python -m uvicorn lux_app:app --host 127.0.0.1 --port
8000`; HTTP GET / returned 200 and /api/health reported generation disabled.
The HTML interface has loading/error/success states, explicit learned/baseline
labels, technical details, and retained session assessments with input and score
comparisons. Browser visual/interaction automation has not been performed.

Initial regression suite: 427 passed in 134.96 seconds. New API tests initially:
6 passed in 10.62 seconds. Further test results are recorded below.
Saved OOF metrics above remain historical, not newly rerun performance estimates.
Prompt SVR has lower MAE than mean but worse MAE than median; selected hidden
and fusion forests have better RMSE/R2 than both constants but worse MAE than
median. Searching many models limits claims; no superiority claim for the live
batch-05 Ridge artifacts follows from those different OOF models.

Final targeted tests: `python -m pytest tests/test_live_app.py -q`: 10 passed,
one Starlette/httpx deprecation warning, 13.29 seconds. Hidden forward/padding
unit test uses mocked tensors and forbids generation; separate actual Qwen fused
integration above is not mocked. Real HTTP POST /api/assess-risk returned the
prompt-only score above. Prompt-edit test uses the actual saved trained artifact
and tokenizer and confirms input features and predictions differ. Browser DOM
state tests and comprehensive malformed hidden-artifact tests remain outstanding.

Full regression run completed: 433 passed, one deprecation warning, 133.45 seconds. That run collected the initial six new tests; the final targeted run includes all ten new tests (four added during the regression run). No failures in either run.

## API validation follow-up (2026-10-28)

Existing .venv activated in PowerShell with process-scoped execution policy
Bypass. No Computer Use was invoked. Python Playwright was not installed
(importlib.find_spec returned None), and neither local node_modules/playwright
nor node_modules/@playwright existed. Therefore this follow-up uses actual HTTP
requests and FastAPI TestClient tests, not browser verification.

Observed live HTTP assessments (same France/Paris input unless noted):
- Prompt-only: HTTP 200, 0.21740319587972134, learned true, no hidden extraction.
- Edited input `Who?` / `Paris.`: HTTP 200, 0.2895665419270655.
- Hidden-only: HTTP 200, 0.07306806743144989, hidden shape [29,2048].
- Fused: HTTP 200, 0.47428810596466064, hidden shape [29,2048].
- Explicit mean: HTTP 200, 0.20864197530864198.
- Explicit median: HTTP 200, 0.0.
- Empty question and unknown predictor: HTTP 422, no risk score.

Full request/response evidence, including predictor identities, feature values,
model revision, pooling, timestamps and artifact hashes, is saved separately in
`docs/engineering/live_api_validation_20261028.json`. Actual hidden/fused requests
used cached frozen Qwen and saved trained preprocessing. Source inspection shows
no generation call in assessment; the mocked forward-only test additionally
fails if generate is invoked. HTTP responses contain no generated answer.
This is implementation evidence, not a measurement of predictive accuracy.

New missing-artifact test redirects the artifact directory to a temporary empty
directory, without moving or altering research artifacts. It requires HTTP 503,
no risk_score, explicit no-fallback message, and no loaded predictor. Backend
failure injection requires sanitized HTTP 503. Unknown selection requires HTTP
422 without a score. No reproducible application defect was found in these
checks. Historical metrics and artifacts were preserved.

Still NOT browser-verified: rendered layout, click/keyboard submission, loading
indicator visibility, actual score-card rendering, session history persistence
and numerical comparison display, dropdown interaction, technical disclosure
expansion, client-side empty-field validation, and browser error/recovery states.
The HTML source implements these paths, but source inspection and independent
API requests do not prove that its JavaScript executes correctly in a browser.
No browser-policy bypass was attempted. These remain explicit validation limits.

Observed follow-up test result: python -m pytest tests/test_live_app.py -q: 12 passed, one Starlette/httpx deprecation warning, 21.42 seconds. GET / returned HTTP 200. No new full-suite run was performed in this follow-up.
