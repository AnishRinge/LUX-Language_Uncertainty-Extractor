"""Check saved OOF joins with small synthetic artifacts."""
import json
import numpy as np
import pytest
from docs.engineering.analyze_saved_oof import analyze


def fixture_files(tmp_path):
    cohort = tmp_path / 'experiments/final_cohort_2026_10_24'
    saved = tmp_path / 'experiments/expanded_model_selection_2026_10_24'
    cohort.mkdir(parents=True)
    saved.mkdir(parents=True)
    np.save(cohort / 'pids.npy', ['a', 'b', 'c', 'd'])
    np.save(cohort / 'y.npy', [0., 1., 0., 1.])
    folds = [{'fold': 0, 'train_pids': ['a', 'b'], 'test_pids': ['c', 'd']},
             {'fold': 1, 'train_pids': ['c', 'd'], 'test_pids': ['a', 'b']}]
    (saved / 'fold_assignments.json').write_text(json.dumps({'fold_assignments': folds}))
    rows = [dict(model='dummy_' + name, config='prompt',
                 oof_pids=['d', 'b', 'a', 'c'], oof_true=[1, 1, 0, 0],
                 oof_predictions=[.5] * 4) for name in ['mean', 'median']]
    path = saved / 'nested_cv_results.json'
    path.write_text(json.dumps({'results_all': rows}))
    return path, rows


def test_analysis_aligns_by_id_and_is_deterministic(tmp_path):
    fixture_files(tmp_path)
    a = analyze(tmp_path)
    assert a == analyze(tmp_path)
    assert a['results']['dummy_mean|prompt']['metrics']['mae'] == .5
    assert a['results']['dummy_mean|prompt']['effects']['median']['paired_bootstrap_95'] == [0, 0]


def test_analysis_rejects_target_mismatch(tmp_path):
    path, rows = fixture_files(tmp_path)
    rows[0]['oof_true'][0] = 0
    path.write_text(json.dumps({'results_all': rows}))
    with pytest.raises(ValueError, match='target mismatch'):
        analyze(tmp_path)


def test_analysis_rejects_fold_overlap(tmp_path):
    path, _ = fixture_files(tmp_path)
    folds = path.parent / 'fold_assignments.json'
    d = json.loads(folds.read_text())
    d['fold_assignments'][0]['train_pids'].append('c')
    folds.write_text(json.dumps(d))
    with pytest.raises(ValueError, match='Contaminated'):
        analyze(tmp_path)


def test_analysis_rejects_incorrect_baseline(tmp_path):
    path, rows = fixture_files(tmp_path)
    rows[0]['oof_predictions'][0] = .8
    path.write_text(json.dumps({'results_all': rows}))
    with pytest.raises(ValueError, match='outer training fold'):
        analyze(tmp_path)
