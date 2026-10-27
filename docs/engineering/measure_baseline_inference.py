"""Baseline-only operational probe; no data, target model, training or evaluator.

Run from repository root:
python -m docs.engineering.measure_baseline_inference docs/engineering/milestone1_measurements.json
Output creation is exclusive. Synthetic texts are a length/encoding workload, not
a representative sample of real-world predictive performance.
"""
import ctypes
from ctypes import wintypes
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time


def sample():
    prompts = []
    for size in [32, 256, 1024, 4096, 16384, 65536]:
        prompts.append(('text_' + str(size), ('What does the passage say? ' * (size // 26 + 1))[:size]))
        prompts.append(('structured_' + str(size), {
            'question': 'पाठ क्या कहता है? What does the passage say?',
            'context': ('A synthetic passage about rivers. ' * (size // 30 + 1))[:size],
        }))
    return prompts


def memory():
    """Windows native per-process counters, bytes, not traced Python allocations."""
    class Counters(ctypes.Structure):
        _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD)] + [
            (name, ctypes.c_size_t) for name in ['PeakWorkingSetSize', 'WorkingSetSize',
                'QuotaPeakPagedPoolUsage', 'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage',
                'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage']]
    data = Counters()
    data.cb = ctypes.sizeof(data)
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    api = ctypes.WinDLL('psapi', use_last_error=True).GetProcessMemoryInfo
    api.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    api.restype = wintypes.BOOL
    if not api(kernel.GetCurrentProcess(), ctypes.byref(data), data.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    return dict(working_set_bytes=data.WorkingSetSize,
                peak_working_set_bytes=data.PeakWorkingSetSize)


def physical_memory():
    class Status(ctypes.Structure):
        _fields_ = [('length', wintypes.DWORD), ('load', wintypes.DWORD)] + [
            (name, ctypes.c_ulonglong) for name in ['total', 'available', 'page_total',
                                                  'page_available', 'virtual_total',
                                                  'virtual_available', 'extended']]
    data = Status()
    data.length = ctypes.sizeof(data)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(data)):
        raise ctypes.WinError()
    return data.total


def distribution(values):
    ordered = sorted(values)
    return dict(n=len(values), mean=statistics.mean(values), median=statistics.median(values),
                p95=ordered[math.ceil(.95 * len(values)) - 1], min=min(values), max=max(values))


def child():
    started = time.perf_counter_ns()
    from src.predictor.inference import Artifact, GENERATION_CONFIG, MODEL, REVISION, Policy, predict_risk
    imported = time.perf_counter_ns()
    artifact = Artifact.load('docs/engineering/baseline_mean_v2.json')
    loaded = time.perf_counter_ns()
    result = predict_risk('What does the synthetic passage say?', MODEL, REVISION,
                          GENERATION_CONFIG, artifact=artifact)
    action = Policy().route(result)
    ended = time.perf_counter_ns()
    print(json.dumps(dict(import_ms=(imported-started)/1e6, artifact_load_ms=(loaded-imported)/1e6,
                          first_inference_and_route_ms=(ended-loaded)/1e6,
                          action=action, risk=result.estimated_risk, memory=memory())))


def run():
    from src.predictor.inference import Artifact, GENERATION_CONFIG, MODEL, REVISION, Policy, predict_risk
    cold = []
    for _ in range(20):
        started = time.perf_counter_ns()
        result = subprocess.run([sys.executable, '-m', 'docs.engineering.measure_baseline_inference', '--child'],
                                capture_output=True, text=True, check=True, timeout=30)
        ended = time.perf_counter_ns()
        row = json.loads(result.stdout)
        row['process_wall_ms'] = (ended-started)/1e6
        if row['action'] != 'escalate' or row['risk'] != .20864197530864198:
            raise ValueError('Unexpected cold prediction')
        cold.append(row)
    artifact = Artifact.load('docs/engineering/baseline_mean_v2.json')
    policy = Policy()
    for _, prompt in sample():
        for _ in range(10):
            predict_risk(prompt, MODEL, REVISION, GENERATION_CONFIG, artifact=artifact)
    before = memory()
    groups = {}
    for name, prompt in sample():
        timings = []
        first = None
        for _ in range(1000):
            started = time.perf_counter_ns()
            result = predict_risk(prompt, MODEL, REVISION, GENERATION_CONFIG, artifact=artifact)
            action = policy.route(result)
            timings.append((time.perf_counter_ns()-started)/1e6)
            if result.estimated_risk != .20864197530864198 or action != 'escalate':
                raise ValueError('Unexpected warm prediction')
            output = (result, action)
            if first is None:
                first = output
            elif output != first:
                raise ValueError('Nondeterministic repeated request')
        groups[name] = dict(chars=len(prompt) if isinstance(prompt, str) else sum(map(len, prompt.values())),
                            milliseconds=distribution(timings))
    return dict(hardware=dict(os=platform.platform(), cpu=os.environ.get('PROCESSOR_IDENTIFIER'),
                              logical_processors=os.cpu_count(), physical_memory_bytes=physical_memory()),
                python=sys.version, executable=sys.executable, sample='12 synthetic cases: six lengths x text/structured Unicode',
                cold_runs=cold,
                cold_summary_ms={key:distribution([r[key] for r in cold]) for key in [
                    'process_wall_ms','import_ms','artifact_load_ms','first_inference_and_route_ms']},
                warm=groups, warm_memory_before=before, warm_memory_after=memory(),
                method='perf_counter_ns; sequential 20 fresh subprocesses; 120 warmup calls then '
                       '1000 calls per case including routing; serialization/assertions excluded from warm timing. '
                       'Process wall includes startup/import/load/first call/memory probe/JSON/stdout/exit. '
                       'OS filesystem cache is not flushed. Memory is native Windows working set and process peak.',
                limitations='Local functional microbenchmark, not production throughput or real prompt distribution. '
                            'No concurrency, target-model loading, hidden-state extraction, generation, evaluator or network. '
                            'Module bootstrap before child timer is included only in process wall.')


if __name__ == '__main__':
    if sys.argv[1] == '--child':
        child()
    else:
        destination = Path(sys.argv[1]).resolve()
        if Path('docs/engineering').resolve() not in destination.parents:
            raise ValueError('Output must be under docs/engineering')
        if destination.exists():
            raise FileExistsError(destination)
        result = run()
        with destination.open('x', encoding='utf-8') as f:
            json.dump(result, f, indent=2, allow_nan=False)
