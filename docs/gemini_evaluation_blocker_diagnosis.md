# LUX Production Evaluation Blocker Diagnosis

This document preserves the initial diagnosis for history. The dated
current-state addendum at the end supersedes its recommendation to make a new
controlled request; an actual request and the exact API errors are now
persisted.

## 1. Project/key configuration

### How GEMINI_API_KEY is loaded and passed to genai.Client

Code references (run_production.py, src/evaluation/gemini_evaluator.py):
- API key read from env var GEMINI_API_KEY (never hardcoded). _load_env_file() in run_production.py scans for .env/.venv/.env etc and materializes GEMINI_API_KEY into os.environ if present.
- GeminiEvaluator stores _resolved_api_key (from param or os.environ) and passes it to genai.Client(api_key=..., http_options with timeout). Key is redacted via _redact() before persistence.
- Client call: client.models.generate_content(model=..., contents=user_text, config={temperature, response_mime_type, response_schema, system_instruction=...})

### Project attribution (from existing evidence)

- Code does not pass project/project_id to genai.Client; SDK uses project associated with the API key by default.
- No project identifier appears in persisted artifacts (evaluations/*.json, generations.json, checkpoint.json, reports). Smoke test report lists evaluator model only.
- Available evidence does NOT establish which Google AI Studio/project received requests. Determining the exact project would require additional API/project metadata not present in existing logs/artifacts.

## 2. Quota errors

Analyzed existing evaluation records in data/production/evaluations (16 files, all SUCCESS). No records contain error_type 'QuotaExhaustedError'. Attempts with retry details show transient 503 UNAVAILABLE (ServerError/TransientEvaluationError) cases that eventually succeeded (e.g. some needed 2-4 attempts; one had 3 retries then success). The only generation marked FAILED in generations.json is squad_v2_56d12bda17492d1400aabb47:2235:1007 (evaluation_status FAILED, no evaluation record file on disk; label/risk null).

- Exact quota ID: not present in existing artifacts.
- Model: gemini-3.8-flash (from evaluator_model in records and config).
- Reported limit: not explicitly recorded in artifacts. The runner report (docs/production_evaluation_runner_report.md, 2026-10-14T20:52:01Z) states stop_reason 'quota_exhausted' for iteration 1 processing gen squad_v2_56d12bda17492d1400aabb47:2235:1007 with error 'QuotaExhaustedError' and attempts=1. However, no persisted evaluation record file exists for that generation; generations.json marks it FAILED but contains no error details beyond status.
- Daily vs per-minute: cannot be definitively separated from existing evidence. Historical 429/quota messages with 'RESOURCE_EXHAUSTED'/'daily' are not found in evaluation records; transient 503s suggest high demand/spikes (likely RPM/availability) rather than daily quota exhaustion in the records we have.
- Project attribution per error: no project field in records; cannot attribute errors to a specific project from existing evidence alone.

## 3. Server errors

From evaluation records (by timestamp): 2026-10-07 has retries (transient 503 UNAVAILABLE) that resolved to SUCCESS; 2026-10-14 also shows mostly SUCCESS with some retries. No 504s observed in stored attempt messages. Generation IDs appear in filenames as squad_v2_<sample>_<dsidx>_<seed>.json.

- 503/504 grouping by timestamp/generation ID: 503 UNAVAILABLE messages appear in attempts for 1001 (2235, 2026-10-14T20:38:22Z had transient then success), 1003 and 1004 (69929, 2026-10-07) had multiple retries then success. No explicit 504 messages found.
- Server availability vs quota failures: transient 503s are classified as retryable/transient (eventually succeeded) per evaluator behavior; quota exhaustion would manifest as RESOURCE_EXHAUSTED/429 quota cases (QuotaExhaustedError) and trigger batch stop. Existing persisted records show server-availability-like transient failures, not quota exhaustion.
- Retry handling impact: GeminiEvaluator uses RetryPolicy (default max_retries=3) with retry_call; on_attempt logs each attempt (status RETRY/FAILED). Transient errors (503/timeout/httpx) are retried; QuotaExhaustedError (429 RESOURCE_EXHAUSTED) is treated as batch-stopping (not retried). Persistent errors (401/403/400/404/ValidationError) are not retried. run_eval_autonomous.py also checks process output/last eval and will stop on quota_exhausted or persistent errors; for service errors (ServerError/ServiceUnavailable/InternalServerError) it stops only if attempts show all failed as RETRIES exhausted. Retries consume additional requests (each attempt is a separate API call) � that�s the intended bounded retry behavior.

## 4. Dashboard discrepancy

Runner report (2026-10-14T20:52:01Z) claims stop_reason=quota_exhausted for gen 2235:1007 with error QuotaExhaustedError (attempts=1). Artifacts:
- generations.json marks squad_v2_56d12bda17492d1400aabb47:2235:1007 as evaluation_status=FAILED; it has no evaluation record file on disk and no stored error details.
- evaluation_runner_log.json is empty/has stop_reason null (log state not preserved with full iteration details in the current file on disk), while the markdown report exists and documents the stop.
- 16 evaluation records on disk (all SUCCESS, some with transient retries) � no QuotaExhaustedError in any persisted record.

Latest screenshot for 'VS Code 3' shows Gemini 3.8 Flash at 1/5 RPM, 790/250K TPM, 1/20 RPD. Comparison with existing timestamps (eval records span 2026-10-07 and 2026-10-14):
- Supported conclusions: there was at least one run where the autonomous runner logged quota_exhausted at ~2026-10-14T20:52:01Z (per report). The screenshot reflects current quota state/limits for 'VS Code 3' context shown. Some evaluations succeeded after transient 503s.
- Unproven: whether the screenshot refers to the same Google project/account as the API key used in these runs; whether the 2026-10-14 stop corresponds to hitting the specific RPD/RPM/TPM numbers shown; whether the failed generation (2235:1007) actually encountered a quota error at runtime vs the runner classifying it as quota_exhausted based on its detection logic at that moment � because no evaluation record file exists for it and the generations.json entry lacks error details. Project identity is not recorded in artifacts.
- Key point: without the exact error payload from the Gemini API (with code/status/message, and the project context) for the failing call, we cannot conclusively map the dashboard numbers to the observed failure. The artifacts are consistent with transient availability causing retries and a possible quota/rate classification at stop time, but they do not contain conclusive proof of which quota bucket was hit or that it was the same project as depicted.

## 5. Runner and resumption

### How run_eval_autonomous.py chooses pending and failed generations
- Loads generations.json; for each gen, checks if eval file exists for generation_id. If exists, reads evaluation record; skips if status is SUCCESS or MANUAL_REVIEW (terminal keep). If exists with status FAILED, counts as 'failed_existing' (eligible for retry) � included in pending list (_eval_requests_from_generations in run_production.py logic is mirrored conceptually by autonomous runner which inspects most recent eval and also relies on generation artifact state).
- Autonomous runner: _count_pending() counts gens with no SUCCESS/MANUAL_REVIEW eval file (or eval file exists but not terminal). It uses EvaluationStatus from src.evaluation.schemas and looks up eval path via run_production._generation_evaluation_path.
- On each iteration, runs 'python run_production.py evaluate --max-evals 1 --spacing 12.0', then inspects most recent eval file (_get_most_recent_eval). Stops based on should_stop_for_error logic (quota/persistent/service exhaustion).

### Preservation of successful evaluations and retry correctness
- run_production.py evaluate: builds pending list by skipping gens whose on-disk eval has SUCCESS or MANUAL_REVIEW (terminal_keep). FAILED on-disk records are eligible for retry and included. After each successful evaluation, writes eval record atomically and updates generations.json (sets evaluation_status/label/claims). The code comments state: 'Previously successful evaluations are never discarded or silently repeated.' (run_production.py ~590-650).
- Autonomous runner: checks initial state � if most recent eval indicates quota_exhausted, it stops immediately without changing existing records. The report states 'Existing Records Preserved: Yes � the runner only appends to or updates eval files; SUCCESS and MANUAL_REVIEW records are never overwritten.' Current on-disk state confirms 16 SUCCESS records preserved.

### Scheduled wakeup implementation
- run_eval_autonomous.py has no cron_create/schedule_wakeup/cron_list/cancel_wakeup calls. It implements a simple loop with MAX_ITERATIONS=5, does one subprocess call per iteration, sleeps 2 seconds between iterations, writes log/report. No code registers/enables a scheduled wakeup/task.
- Task state: evaluation_runner_log.json is effectively empty (started_at empty, iterations empty, stop_reason null) while the markdown report documents a prior stop. No evidence of a scheduled cron/wakeup being registered. The 'Next Action' in report says 're-run: python run_eval_autonomous.py' (manual).
- Conclusion: the scheduled wakeup implementation does NOT exist in the codebase as written; it is not actually registered, enabled, or capable of launching the runner. Any claim that it is scheduled is not supported by code or task state.

## 6. Final diagnosis

### Timeline of failures (from artifacts)
1. 2026-10-07: Evaluations run with transient 503 UNAVAILABLE; some require 2-4 attempts but all succeed. Records created.
2. 2026-10-14: More evaluations run; mostly succeed, some with retries (1 retry on 1001 at 20:38:22Z). Later (around 20:46-20:47Z) several SUCCESS with 1 attempt. Report generated 2026-10-14T20:52:01Z states iteration attempted evaluation of gen 2235:1007, stop reason quota_exhausted, error QuotaExhaustedError (attempts=1).
3. Post-stop state: generations.json has 2235:1007 marked evaluation_status=FAILED (no eval file). 16 SUCCESS eval records persist; 297 generations remain without evaluation_status set (pending per generations.json), 1 marked FAILED.

### Quota vs server errors (table)

| Category | Evidence in artifacts | Count/instances | Notes |
|--------|----------------------|-----------------|-------|
| Server availability (503/UNAVAILABLE) | TransientEvaluationError with 503 UNAVAILABLE in attempt messages | Multiple retries across records (e.g. 1003 had 3 retries then success, 1004 had 2 retries then success) | Eventually resolved; treated as retryable. |
| Quota exhaustion (429 RESOURCE_EXHAUSTED) | No QuotaExhaustedError in any persisted eval record; runner report claims it for a gen with no eval file | 0 confirmed in stored records | Claimed at stop time but not corroborated by persisted eval record for that call. |
| Persistent errors (401/403/400/404) | None observed | 0 | N/A |
| Service errors (504) | None observed in messages | 0 | N/A |

### Most likely explanation (hypothesis)

**Hypothesis (unproven):** The autonomous runner stopped at 2026-10-14T20:52:01Z due to detecting quota exhaustion for the attempted evaluation of squad_v2_56d12bda17492d1400aabb47:2235:1007. However, the lack of a persisted evaluation record for that call and the absence of QuotaExhaustedError details in any stored record mean we cannot conclusively distinguish between (a) a genuine quota/rate limit hit (RPM/RPD/TPM) on that specific call, (b) the runner's detection logic/classification at stop time (possibly based on process stderr/most-recent-eval state) leading to quota_exhausted classification, or (c) a transient classification conflated with prior high-demand behavior. The dashboard screenshot for 'VS Code 3' may or may not correspond to the same project as the API key used.

### Evidence needed to conclusively distinguish possibilities
1. **API error payload**: Capture the exact exception/response from the Gemini API for the failing call (status code, error.code, error.status/message, retryInfo) � including whether it was RESOURCE_EXHAUSTED (429) with quota details. This requires running with logging that preserves the raw API error (without exposing the key). The current evaluator redacts but does not persist full API response bodies.
2. **Project context**: Determine which Google AI Studio/project the API key is associated with (and whether it matches 'VS Code 3' in the screenshot). Not present in artifacts.
3. **Runner execution context**: Preserve evaluation_runner_log.json with full iteration details (stdout/stderr tails, the actual eval record content produced, timestamps) at the moment of stop. Current log on disk is truncated/empty relative to the report.
4. **Quota bucket timing**: Correlate failure timestamp (if captured precisely) with the dashboard's RPM/RPD/TPM windows for that project. The report gives 2026-10-14T20:52:01Z as generated time; more precise call timestamp needed.
5. **Reproducible test**: Attempt a single controlled evaluation (max-evals=1) with verbose logging and capture the full error; if it fails with quota, compare against dashboard. If it fails with 503, compare. This is minimal API usage.

### Historical next-action recommendation

The single-call diagnostic recommendation below was made before the
2026-10-17 production attempt. It has been superseded by the persisted
one-generation evaluation and its ten credential attempts; do not repeat it
without new evidence that the retired model or quota state has changed.

---
Generated by diagnostic-only investigation (no code/data modifications).

## Current-state addendum (2026-10-17 19:55 +05:30)

The configured environment contains ten Gemini credential slots; only their
slot names were inspected. The existing primary-model directory contains 19
records for `gemini-2.5-flash` / `CP4.4.1`: 18 `SUCCESS` records (8
`RELIABLE`, 10 `UNRELIABLE`) and one `FAILED` record. The failure is generation
`squad_v2_56dfbfff7aa994140058e111:14320:1005`. Its primary slot returned 429
`RESOURCE_EXHAUSTED`; the next slots returned 404 `NOT_FOUND` with the API
message: “This model models/gemini-2.5-flash is no longer available to new
users.” That same global retirement response was persisted for slots 2–10.
The previous classifier incorrectly treated slots 4–10 as credential-specific
model access failures and continued failover. This explains why no new
controlled request was made: the actual full credential-bound attempt already
demonstrated a global model blocker, and retrying would be futile.

The evaluator now classifies a 404 retirement/unknown-model response as
`GlobalModelConfigurationError`, which stops credential failover; genuine
credential-specific 401/403 errors can still fail over. Mocked tests cover
both paths.

The 18 resolved primary records cover only 1 of 31 complete prompt groups
(risk 1.0); 291 generations have no primary record and one has a failed
record. The version-filtered partial target artifact is
`data/production/gemini_2_5_flash_CP4_4_1/targets.json`. This is insufficient
for a train/test predictor comparison. No additional Gemini 2.5 calls or
predictors were run during this continuation. Separately,
CP4.5.0 remains at 84/310 records (82 success, one manual review, one failed),
with 226 missing and its ten slots quota-exhausted; no worker is running.

On the 17 same-generation records successfully resolved by both
`gemini-2.5-flash` / `CP4.4.1` and `gemini-3.8-flash` / `CP4.5.0`, labels
agree in all 17 cases (8 `RELIABLE`, 9 `UNRELIABLE`). This small selected
overlap is descriptive agreement only, not independent ground truth and not
permission to merge evaluator versions. Pairwise provenance is recorded in
`data/production/evaluator_agreement/gemini_2_5_flash_CP4_4_1__gemini_3_8_flash_CP4_5_0.json`.

## Evaluator availability update (2026-10-17 20:14 +05:30)

The model/request evidence is:

| Model | Persisted/request evidence | Decision |
|---|---|---|
| `gemini-2.5-flash` | Historical successful records exist, but a later request returned 404 `NOT_FOUND`: “no longer available to new users.” Other credential slots received the same global message; the primary slot also had a 429 quota error. | Retired for this API access; do not retry via credential rotation. |
| `gemini-3.8-flash` | 82 successful CP4.5.0 records; subsequently all ten slots had 429 quota errors. The persisted metric is `generate_content_free_tier_requests`, limit 5 for this model. | Provider was accessible, but saved state marks all ten slots exhausted. No new calls made. |
| `gemini-2.5-flash-lite` | Exactly one real generation evaluation using `GEMINI_API_KEY`, with retries and failover disabled, returned 404 `NOT_FOUND`: “no longer available to new users.” The API suggested `gemini-3.5-flash-lite`. | Globally unavailable to these credentials; no retry through other keys. The suggested model was not in the requested candidate list and was not tried. |

No persisted response or project dashboard evidence gives current numeric
RPM or TPM availability. The Gemini 3.8 error identifies a free-tier request
metric and limit but does not establish all live quota windows; a standalone
quota check was not made. There were no HTTP 401/403 responses in the saved
records inspected. The single Flash-Lite API call reached the provider and
failed before producing structured evaluation output; there are zero resolved
CP4.6.0 labels.

The isolated CP4.6.0 artifacts are under
`data/production/cp4_6_0_gemini_2_5_flash_lite/`: one failed evaluation record,
one run log, and targets showing zero eligible prompts / 310 missing or
unresolved outcomes. The CP4.5.0 checkpoint remains unchanged at 84 records.
No provider worker or monitor is running, and no further calls were made.
