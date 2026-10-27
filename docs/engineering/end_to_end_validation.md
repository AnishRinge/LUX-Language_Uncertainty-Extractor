# LUX Milestone 1: end-to-end validation

## Verdict

**PASS for the supported baseline-only Python interface.** Functional inference,
explicit artifact selection/fallback, error boundaries and conservative routing
passed integration checks. This is not certification of prompt-specific accuracy,
calibration, production capacity or downstream hallucination reduction. Cold import
overhead is a material operational limitation, recorded below.

## Exact public interface tested

```python
from src.predictor.inference import (
    Artifact, GENERATION_CONFIG, MODEL, REVISION, Policy, predict_risk,
)
baseline = Artifact.load('docs/engineering/baseline_mean_v2.json')
prediction = predict_risk(
    {'question': 'Why?', 'context': 'A synthetic passage.'},
    MODEL, REVISION, dict(GENERATION_CONFIG), artifact=baseline,
)
payload = prediction.to_dict()
action = Policy().route(prediction)  # 'escalate'
```

The target is Qwen/Qwen3-1.7B at revision
70d244cc86ccca08cf5af4e1e306ecf908b1ad5e. Exact supported configuration is
device=cpu, thinking_mode=False, do_sample=True, temperature=0.7, top_p=0.8,
top_k=20, max_new_tokens=512. Numeric/boolean types are validated strictly.

The shipped mean returns .20864197530864198; the separately loaded median returns
0. Both are explicit constant baselines, not individualized estimates. Returned
schema: estimated_risk, predictor_id, target_model, model_revision,
feature_pipeline_version, baseline_statistic, calibrated, uncertainty, warnings.
Risk is finite and bounded; calibrated=False and uncertainty=None. JSON serialization
was checked with allow_nan=False. Versioned pipeline is constant-no-features-v1.

Routing exposes Policy(verify_at=None, escalate_at=None, allow_uncalibrated=False)
and Policy.route(prediction). No new operational thresholds were introduced. Existing
test-only policy values exercise invalid-risk fail-safe behavior. Default routing
escalates, including for the zero median. There is no API server or action executor.

Fallback is application-controlled: Artifact.load raises for an unavailable preferred
artifact; the caller may explicitly load a known baseline or route None to escalation.
No automatic estimator substitution exists. The integration test checks the selected
baseline statistic, pipeline version, fingerprint and structured log record. It does
not imply a learned prediction path was exercised.

## Coverage and changes

Only new Milestone 1 files were created:

- tests/test_inference_end_to_end.py: 49 integration cases covering both shipped v2
  artifacts, exact output schema, model/config rejection, missing/corrupt/incompatible
  artifacts, invalid encoding and duplicate/missing/extra JSON fields, prompt boundaries,
  explicit fallback, repeated requests, no mutation and fail-safe invalid-risk routing.
- docs/engineering/measure_baseline_inference.py: reproducible Windows baseline-only
  probe using standard-library timers, subprocesses and native process memory counters.
- milestone1_measurements.json and milestone1_measurements_v2.json: actual operational
  outputs; v2 includes the final warm expected-value assertion and is authoritative.
- milestone1_preservation_before.json: file size/mtime inventory of data and experiments.
- This report.

Post-generation fields (answers, labels, evaluations, risk targets, answerability) and
hidden-state payload/reference injection are rejected as extra input fields. This
pipeline has no feature extractor and does not consume any prompt features; no failed
hidden-state extraction can become a zero vector. Text can contain arbitrary semantic
content, but none is converted into predictor features by this constant implementation.

No defect was found in src/predictor/inference.py and it was not modified. The first
test run exposed a Windows test-harness issue: pytest used a 100,000-character input
as a test ID, causing fixture setup errors (45 passed, 2 errors). Short explicit IDs
fixed the harness. A fresh review found no important issue and suggested checking
the expected warm risk/action in addition to repeatability; that assertion was added
outside the timed region, and all 12 workload cases are also checked by pytest.

## Commands and actual results

Every shell used process-only execution-policy bypass followed by activation; no
machine policy was changed:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -ErrorAction Stop
. .\.venv\Scripts\Activate.ps1
python -c "import pathlib,sys; print(sys.executable); assert pathlib.Path(sys.executable).resolve()==pathlib.Path('.venv/Scripts/python.exe').resolve()"
```

Interpreter verified as this project's .venv/Scripts/python.exe. Initial read-only
inspection covered the implementation report, audit, README, public API, existing
tests and git status. The working tree was already dirty and was preserved.

Executed test commands:

```powershell
python -m pytest tests/test_inference_end_to_end.py -q
python -m pytest tests/test_inference_end_to_end.py tests/test_risk_inference.py tests/test_saved_oof_analysis.py tests/test_baseline_export.py -q --tb=short
python -m pytest tests -q --tb=short
```

Results: initial harness failure above; then 73 focused tests passed in 11.41 s;
after two additional cases, 75 passed in 16.01 s. The final focused run after review
passed **75 tests in 15.97 s**. The broader suite passed **409 tests in 273.07 s**
at its collection snapshot; the two later test additions were included in the final
focused run. No broader failures occurred. Existing regression tests use synthetic
fixtures; no production model was trained or research experiment rerun.

Operational commands (each writes a new file exclusively):

```powershell
python -m docs.engineering.measure_baseline_inference docs/engineering/milestone1_measurements.json
python -m docs.engineering.measure_baseline_inference docs/engineering/milestone1_measurements_v2.json
git diff --check
git diff --stat
```

Both probes completed successfully. The second repeats the same timing boundaries
with the final expected-value assertion. No generation, evaluator, search, model
training or download is invoked by the probe.

## Operational method and limitations

Twenty fresh Python subprocesses separately measure module import, artifact load,
and first inference plus routing. Parent wall time includes process startup, imports,
load, prediction, memory query, output serialization and exit. These are process-cold
starts: filesystem caches were not flushed, so they are not disk-cold tests.

Warm workload is 12 documented synthetic cases: six text lengths (32, 256, 1,024,
4,096, 16,384, 65,536 characters), each as rendered text and as question/context.
Structured questions contain Unicode Hindi and English; the context has the stated
length. This represents supported lengths/encodings, not the distribution of real
SQuAD or application traffic. Each case has 10 warmup calls and 1,000 measured calls:
120 warmups and 12,000 measured sequential requests per probe. Repeated outputs must
remain equal. The final probe also checks the expected risk/action on every request.

Timers use perf_counter_ns. Warm timings include public inference plus routing,
including per-call artifact validation, and exclude output serialization and assertions.
Logging uses default configuration (no INFO handler enabled). Native Windows
GetProcessMemoryInfo reports current and peak working set for the entire process,
including Python and imported dependencies, not Python heap alone. Before/after warm
working sets are a bounded repetition check, not proof against long-running leaks.

Hardware: Windows build 10.0.26200, Intel64 Family 6 Model 140 Stepping 1,
GenuineIntel, 8 logical processors, 16,952,647,680 bytes physical RAM (about 15.79 GiB).
Get-CimInstance Win32_Processor was attempted for the marketing CPU name but returned
Access denied (HRESULT 0x80041003); no policy/permission change was attempted.
Full CPU model name and physical core count therefore remain unavailable.

The full regression run overlapped the operational probes, creating CPU/I/O contention.
These measurements characterize this local session, not isolated production latency.
No concurrency/load test, service/network overhead, target-model load, hidden-state
extraction, generation or verification latency was measured. No production SLO exists
against which to certify latency. A bounded probe cannot establish production capacity.

### Actual measurements (final v2 probe)

All timings below are milliseconds. Cold results use 20 successful fresh processes;
p95 is the nearest-rank order statistic, not a confidence bound.

| Stage | Mean | Median | p95 | Range |
|---|---:|---:|---:|---:|
| Entire cold process | 6138.483 | 4266.744 | 16010.238 | 2674.676–16538.095 |
| Predictor module import | 5572.580 | 3808.323 | 15152.033 | 2334.639–15789.683 |
| JSON artifact load | 0.880 | 0.692 | 1.821 | 0.419–2.129 |
| First inference + routing | 0.235 | 0.209 | 0.373 | 0.115–0.406 |

The "load" stage loads only the small baseline JSON artifact; no target model is
loaded. Import time includes the eagerly loaded legacy predictor dependencies.

Each warm row contains 1,000 successful deterministic inference/routing calls:

| Synthetic case | Mean ms | Median ms | p95 ms |
|---|---:|---:|---:|
| Text 32 | 0.0948 | 0.0907 | 0.1300 |
| Structured context 32 | 0.0961 | 0.0920 | 0.1159 |
| Text 256 | 0.0941 | 0.0901 | 0.1092 |
| Structured context 256 | 0.1097 | 0.0988 | 0.1579 |
| Text 1024 | 0.0977 | 0.0984 | 0.1272 |
| Structured context 1024 | 0.0966 | 0.0978 | 0.1150 |
| Text 4096 | 0.0963 | 0.0933 | 0.1354 |
| Structured context 4096 | 0.0985 | 0.0946 | 0.1245 |
| Text 16384 | 0.0960 | 0.0912 | 0.1297 |
| Structured context 16384 | 0.0975 | 0.0917 | 0.1250 |
| Text 65536 | 0.0991 | 0.0922 | 0.1332 |
| Structured context 65536 | 0.0899 | 0.0910 | 0.1282 |

Parent working set before/after the warm workload: 122,236,928 / 122,593,280 bytes
(increase 356,352 bytes). Final process peak working set: 122,593,280 bytes, about
116.9 MiB. Fresh child peak working sets ranged 121,511,936–122,314,752 bytes.
No request failed and every final-probe result matched mean risk and escalation.

The retained initial probe measured cold median 7808.830 ms and warm case means
0.1447–0.2071 ms. The difference between probes underscores scheduling/cache/contention
variability; the v2 numbers are the final checked implementation, not a claim of a
performance improvement. Raw results and all case ranges are in the two JSON files.

## Preservation and remaining limitations

Before/after inventories cover 10,487 files under data/ and experiments/: zero files
added, removed or changed by size/mtime. Both pinned audited target/generation SHA256
hashes still match. Metadata checks are not cryptographic checks of every historical
byte. Existing engineering artifacts, datasets, predictions, folds and checkpoints
were not written. Git diff --check passed; tracked diffs include pre-existing user
work, and the new milestone files were inspected separately because they are untracked.
No files were staged, committed or pushed.

Cold import eagerly loads the existing predictor package and its scikit-learn stack
through src/predictor/__init__.py, even for a constant predictor. This is observed
startup/dependency overhead, not target-model loading. It was left unchanged to avoid
an unrelated package API refactor; lazy legacy exports can be evaluated separately.

No individual calibration, uncertainty, OOD detector, learned-model fallback, learned
hidden-state inference, cross-model support or downstream reliability gain was tested
or claimed because those capabilities are unsupported. Human validation was not
required. Next operational step: measure startup and sustained concurrent traffic in
the intended deployment environment, with a latency budget, before deployment claims.
