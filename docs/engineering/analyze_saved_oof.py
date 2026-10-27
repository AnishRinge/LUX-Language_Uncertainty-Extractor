"""Read-only historical analysis; output exclusively to a NEW engineering file.

Run from repository root: python docs/engineering/analyze_saved_oof.py OUTPUT.json
Bootstrap intervals condition on existing folds/models; no model refitting occurs.
"""
import json
from pathlib import Path
import sys

import numpy as np


def metrics(y, pred):
    residual = y - pred
    denom = np.sum((y - y.mean()) ** 2)
    return dict(mae=float(np.abs(residual).mean()),
                rmse=float(np.sqrt(np.mean(residual ** 2))),
                r2=float(1 - np.sum(residual ** 2) / denom) if denom else None)


def analyze(root):
    cohort = root / 'experiments/final_cohort_2026_10_24'
    saved = root / 'experiments/expanded_model_selection_2026_10_24'
    ids = np.load(cohort / 'pids.npy').astype(str).tolist()
    y = np.load(cohort / 'y.npy')
    if len(set(ids)) != len(ids) or y.shape != (len(ids),) or not np.isfinite(y).all():
        raise ValueError('Invalid cohort alignment')
    position = {pid: i for i, pid in enumerate(ids)}
    folds = json.loads((saved / 'fold_assignments.json').read_text())['fold_assignments']
    tests = []
    for fold in folds:
        tr, te = fold['train_pids'], fold['test_pids']
        if len(set(tr)) != len(tr) or len(set(te)) != len(te) or set(tr) & set(te) or set(tr) | set(te) != set(ids):
            raise ValueError('Contaminated or incomplete fold')
        tests.extend(te)
    if len(tests) != len(ids) or set(tests) != set(ids):
        raise ValueError('Every prompt must be tested exactly once')
    results = json.loads((saved / 'nested_cv_results.json').read_text())['results_all']
    aligned = {}
    for row in results:
        pids = row['oof_pids']
        if len(pids) != len(ids) or set(pids) != set(ids):
            raise ValueError('OOF ID mismatch')
        true, pred = np.asarray(row['oof_true']), np.asarray(row['oof_predictions'])
        if true.shape != y.shape or pred.shape != y.shape or not np.isfinite(pred).all():
            raise ValueError('Invalid OOF values')
        order = np.asarray([position[pid] for pid in pids])
        if not np.array_equal(true, y[order]):
            raise ValueError('OOF target mismatch')
        values = np.empty_like(y)
        values[order] = pred
        key = row['model'] + '|' + row['config']
        if key in aligned:
            raise ValueError('Duplicate result configuration')
        aligned[key] = values
    for name in ('mean', 'median'):
        baseline = aligned['dummy_' + name + '|prompt']
        for fold in folds:
            train = [position[pid] for pid in fold['train_pids']]
            test = [position[pid] for pid in fold['test_pids']]
            expected = np.mean(y[train]) if name == 'mean' else np.median(y[train])
            if not np.allclose(baseline[test], expected, rtol=0, atol=1e-12):
                raise ValueError('Baseline is not fitted on outer training fold')
    rng = np.random.default_rng(20261027)
    bootstrap = rng.integers(0, len(y), size=(2000, len(y)))
    report = {}
    for key, pred in aligned.items():
        fold_metrics = []
        for fold in folds:
            idx = [position[pid] for pid in fold['test_pids']]
            fold_metrics.append(dict(fold=fold['fold'], n=len(idx), **metrics(y[idx], pred[idx])))
        effects = {}
        for name in ('mean', 'median'):
            base = aligned['dummy_' + name + '|prompt']
            # Positive gain means lower absolute error than the baseline.
            gain = np.abs(y - base) - np.abs(y - pred)
            ci = np.quantile(gain[bootstrap].mean(axis=1), [.025, .975])
            base_mae = np.abs(y - base).mean()
            effects[name] = dict(mae_gain=float(gain.mean()),
                                 relative_mae_gain=float(gain.mean() / base_mae),
                                 paired_bootstrap_95=ci.tolist())
        report[key] = dict(metrics=metrics(y, pred), folds=fold_metrics, effects=effects,
                           out_of_bounds=int(((pred < 0) | (pred > 1)).sum()))
    return dict(seed=20261027, bootstrap_replicates=2000, n=len(y), results=report,
                limitations='Conditional prompt bootstrap, not nested-CV retraining uncertainty; '
                'shared contexts and model-selection multiplicity are not accounted for. '
                'No calibration or individual uncertainty claims; predictions are not clipped.')


if __name__ == '__main__':
    output = Path(sys.argv[1]).resolve()
    destination = Path('docs/engineering').resolve()
    if destination not in output.parents:
        raise ValueError('Output must be a new file under docs/engineering')
    result = analyze(Path('.'))
    with output.open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2, allow_nan=False)
