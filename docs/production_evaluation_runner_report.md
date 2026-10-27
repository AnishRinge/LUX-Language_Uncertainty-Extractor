# Production Evaluation Runner Report

Generated: 2026-10-17T10:21:55.586871+00:00

## Session Budget
- Configured budget: 9000s
- Actual elapsed: 15.7s

## Interpreter / Dependency
- Runner interpreter: E:\Projects\LLM-Hallucination-Prediction\LUX-Language_Uncertainty-Extractor\.venv\Scripts\python.exe
- Subprocess Python (venv): E:\Projects\LLM-Hallucination-Prediction\LUX-Language_Uncertainty-Extractor\.venv\Scripts\python.exe

## Starting Counts
- SUCCESS: 34
- FAILED: 0
- MANUAL_REVIEW: 0
- UNKNOWN: 0

## Run Summary
- Iterations (evaluations attempted): 1
- Stop reason: persistent_error
- Process at report generation: stopped
- Newly resolved evaluations: 0
- Newly failed evaluations: 1
- Newly marked manual review: 0

## End-of-Run Counts
- SUCCESS: 34
- FAILED: 1
- MANUAL_REVIEW: 0
- UNKNOWN: 0
- Remaining pending: 276

## Progress Log
- [2026-10-17T08:13:26.819336+00:00] Gen squad_v2_56d12bda17492d1400aabb47:2235:1009: status=SUCCESS, error=None, attempts this run=1, total recorded attempts=1, credential slots=not recorded, exit code=0
- [2026-10-17T08:13:44.178664+00:00] Gen squad_v2_56d12bda17492d1400aabb47:2235:1010: status=SUCCESS, error=None, attempts this run=1, total recorded attempts=1, credential slots=not recorded, exit code=0
- [2026-10-17T08:13:58.159609+00:00] Gen squad_v2_56d6328b1c85041400946fea:6859:1001: status=SUCCESS, error=None, attempts this run=1, total recorded attempts=1, credential slots=not recorded, exit code=0
- [2026-10-17T08:14:11.859759+00:00] Gen squad_v2_56d6328b1c85041400946fea:6859:1002: status=SUCCESS, error=None, attempts this run=1, total recorded attempts=1, credential slots=not recorded, exit code=0
- [2026-10-17T08:14:27.778692+00:00] Gen squad_v2_56d6328b1c85041400946fea:6859:1003: status=SUCCESS, error=None, attempts this run=1, total recorded attempts=1, credential slots=not recorded, exit code=0
- [2026-10-17T08:14:42.905414+00:00] Gen squad_v2_56d6328b1c85041400946fea:6859:1004: status=SUCCESS, error=None, attempts this run=1, total recorded attempts=1, credential slots=not recorded, exit code=0
- [2026-10-17T08:15:00.053153+00:00] Gen squad_v2_56d6328b1c85041400946fea:6859:1005: status=SUCCESS, error=None, attempts this run=2, total recorded attempts=2, credential slots=not recorded, exit code=0
- [2026-10-17T08:15:11.306463+00:00] Gen squad_v2_56d6328b1c85041400946fea:6859:1006: status=SUCCESS, error=None, attempts this run=1, total recorded attempts=1, credential slots=not recorded, exit code=0
- [2026-10-17T08:15:24.114780+00:00] Gen squad_v2_56d6328b1c85041400946fea:6859:1007: status=SUCCESS, error=None, attempts this run=1, total recorded attempts=1, credential slots=not recorded, exit code=0
- [2026-10-17T08:15:35.571491+00:00] Gen squad_v2_56d6328b1c85041400946fea:6859:1008: status=SUCCESS, error=None, attempts this run=1, total recorded attempts=1, credential slots=not recorded, exit code=0
- [2026-10-17T08:15:47.183371+00:00] Gen squad_v2_56d6328b1c85041400946fea:6859:1009: status=SUCCESS, error=None, attempts this run=1, total recorded attempts=1, credential slots=not recorded, exit code=0
- [2026-10-17T08:15:59.261741+00:00] Gen squad_v2_56d6328b1c85041400946fea:6859:1010: status=SUCCESS, error=None, attempts this run=1, total recorded attempts=1, credential slots=not recorded, exit code=0
- [2026-10-17T08:16:09.292868+00:00] Gen squad_v2_56dfbfff7aa994140058e111:14320:1001: status=SUCCESS, error=None, attempts this run=1, total recorded attempts=1, credential slots=not recorded, exit code=0
- [2026-10-17T08:16:20.753427+00:00] Gen squad_v2_56dfbfff7aa994140058e111:14320:1002: status=SUCCESS, error=None, attempts this run=1, total recorded attempts=1, credential slots=not recorded, exit code=0
- [2026-10-17T08:16:32.031579+00:00] Gen squad_v2_56dfbfff7aa994140058e111:14320:1003: status=SUCCESS, error=None, attempts this run=1, total recorded attempts=1, credential slots=not recorded, exit code=0
- [2026-10-17T08:16:44.276640+00:00] Gen squad_v2_56dfbfff7aa994140058e111:14320:1004: status=SUCCESS, error=None, attempts this run=1, total recorded attempts=1, credential slots=not recorded, exit code=0
- [2026-10-17T08:16:52.649195+00:00] Gen squad_v2_56dfbfff7aa994140058e111:14320:1005: status=FAILED, error=QuotaExhaustedError, attempts this run=1, total recorded attempts=1, credential slots=not recorded, exit code=0
- [2026-10-17T09:24:03.757539+00:00] Gen squad_v2_56dfbfff7aa994140058e111:14320:1005: status=FAILED, error=PersistentEvaluationError, attempts this run=1, total recorded attempts=2, credential slots=GEMINI_API_KEY, GEMINI_API_KEY_2, exit code=0
- [2026-10-17T09:29:16.760047+00:00] Gen squad_v2_56dfbfff7aa994140058e111:14320:1005: status=FAILED, error=PersistentEvaluationError, attempts this run=1, total recorded attempts=3, credential slots=GEMINI_API_KEY_3, exit code=0
- [2026-10-17T10:15:42.368735+00:00] Gen squad_v2_56dfbfff7aa994140058e111:14320:1005: status=FAILED, error=CredentialAccessError, attempts this run=7, total recorded attempts=10, credential slots=GEMINI_API_KEY_10, GEMINI_API_KEY_4, GEMINI_API_KEY_5, GEMINI_API_KEY_6, GEMINI_API_KEY_7, GEMINI_API_KEY_8, GEMINI_API_KEY_9, exit code=0

## Latest Credential Outcomes
- Generation: squad_v2_56dfbfff7aa994140058e111:14320:1005
- Status: FAILED
- Label: none
- GEMINI_API_KEY: daily quota exhausted (HTTP 429)
- GEMINI_API_KEY_2: model access denied for this project (HTTP 404)
- GEMINI_API_KEY_3: model access denied for this project (HTTP 404)
- GEMINI_API_KEY_4: model access denied for this project (HTTP 404)
- GEMINI_API_KEY_5: model access denied for this project (HTTP 404)
- GEMINI_API_KEY_6: model access denied for this project (HTTP 404)
- GEMINI_API_KEY_7: model access denied for this project (HTTP 404)
- GEMINI_API_KEY_8: model access denied for this project (HTTP 404)
- GEMINI_API_KEY_9: model access denied for this project (HTTP 404)
- GEMINI_API_KEY_10: model access denied for this project (HTTP 404)

## Quota / Service Status
- Stopped due to: persistent_error

## Artifact Paths
- Progress log: `data/production/evaluation_runner_log.json`
- This report: `docs/production_evaluation_runner_report.md`
- Evaluation records: `data/production/evaluations/`
- Generation artifacts: `data/production/generations.json`

## Existing Records Preserved
Yes — the runner only appends to or updates eval files; SUCCESS and MANUAL_REVIEW records are never overwritten.

## Next Action
A persistent error was encountered. Inspect the eval record at:
  data/production/evaluations/
Do not re-run automatically.
