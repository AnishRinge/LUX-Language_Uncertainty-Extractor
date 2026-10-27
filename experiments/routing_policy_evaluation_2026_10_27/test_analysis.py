import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('lux_routing_analysis', Path(__file__).with_name('analysis.py'))
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def test_metric_denominators_and_hypothetical_assumptions():
    m = module.metrics(np.array([0., .5, 1., 0.]), np.array([False, False, True, True]))
    assert m['normal_empirical_risk'] == .25
    assert m['normal_any_unreliable_prompt_fraction'] == .5
    assert m['escalation_fraction'] == .5
    assert m['captured_unreliable_fraction'] == pytest.approx(2/3)
    assert m['false_reassurance_generation_proxy'] == .25
    assert m['hypothetical_remaining_per_original_prompt']['0.0'] == .125
    assert m['hypothetical_remaining_per_original_prompt']['1.0'] == .375
    assert module.metrics(np.zeros(2), np.ones(2, dtype=bool))['normal_empirical_risk'] is None
    assert module.metrics(np.zeros(2), np.zeros(2, dtype=bool))['captured_unreliable_fraction'] is None


@pytest.mark.parametrize('threshold', [-.1, 1.1, float('nan'), True, None])
def test_threshold_validation(threshold):
    with pytest.raises(ValueError):
        module.ThresholdRule(threshold)


def test_threshold_boundaries_and_action_config():
    rule = module.ThresholdRule(.2, high_action='escalate')
    assert rule.route(np.array([0., .2, 1.])).tolist() == ['allow', 'escalate', 'escalate']
    assert module.ThresholdRule(0).route(np.array([0.])).tolist() == ['verify']
    with pytest.raises(ValueError):
        rule.route(np.array([float('nan')]))
    with pytest.raises(ValueError):
        rule.route(np.array([-.01]))
    with pytest.raises(ValueError):
        module.ThresholdRule(.2, high_action='allow')


def test_alignment_reorders_and_rejects_duplicates_or_wrong_targets():
    row = dict(oof_pids=['b', 'a'], oof_true=[1., 0.], oof_predictions=[.8, .1])
    assert module.align(row, ['a', 'b'], np.array([0., 1.])).tolist() == [.1, .8]
    with pytest.raises(ValueError):
        module.align({**row, 'oof_pids': ['a', 'a']}, ['a', 'b'], np.array([0., 1.]))
    with pytest.raises(ValueError):
        module.align({**row, 'oof_true': [0., 1.]}, ['a', 'b'], np.array([0., 1.]))


def test_folds_require_disjoint_exact_once_coverage():
    folds = [dict(fold=0, train_pids=['b'], test_pids=['a']),
             dict(fold=1, train_pids=['a'], test_pids=['b'])]
    assert module.validate_folds(folds, ['a', 'b']) == [[0], [1]]
    with pytest.raises(ValueError):
        module.validate_folds([folds[0], folds[0]], ['a', 'b'])
    with pytest.raises(ValueError):
        module.validate_folds([{**folds[0], 'train_pids': ['a', 'b']}, folds[1]], ['a', 'b'])


def test_other_outer_oof_is_not_safe_selection_data():
    with pytest.raises(ValueError, match='lineage'):
        module.require_selection_evidence(['a'], ['b'], None)
    with pytest.raises(ValueError, match='contamination'):
        module.require_selection_evidence(['a'], ['b'], {'a': ['b', 'c']})
    module.require_selection_evidence(['a'], ['b'], {'a': ['c']})


def test_hash_reference_exact_budget_no_targets_and_order_invariance():
    ids = ['a', 'b', 'c', 'd']
    mask = module.hash_mask(ids, [[0, 1], [2, 3]], [1, 1])
    assert mask.sum() == 2
    assert mask[:2].sum() == mask[2:].sum() == 1
    reverse = module.hash_mask(ids[::-1], [[0, 1], [2, 3]], [1, 1])
    assert {p for p, selected in zip(ids, mask) if selected} == {p for p, selected in zip(ids[::-1], reverse) if selected}


def test_paired_bootstrap_identical_policy_zero_and_missing_support():
    y = np.array([0., 1., .5, 0.])
    mask = np.array([False, True, False, True])
    idx = module.bootstrap_indices([[0, 1], [2, 3]], 100, 42)
    result = module.paired_intervals(y, mask, mask, idx)
    assert result['escalation_fraction']['interval_95'] == [0., 0.]
    result = module.paired_intervals(y, np.ones(4, dtype=bool), mask, idx)
    assert result['normal_empirical_risk']['interval_95'] is None


def test_costs_are_explicit_not_selected():
    cost = module.Costs(unreliable=10, verify=1, escalate=2)
    assert module.expected_cost(np.array([.1, .9]), np.array(['allow', 'verify']), cost, 0) == 1
    with pytest.raises(ValueError):
        module.Costs(unreliable=-1, verify=1, escalate=2)


def make_sources(root):
    import hashlib
    import json
    targets = root/'data/production/cp4_7_0_gemini_3_5_flash_lite/targets_batch_05.json'
    generations = root/'data/production/squad_v2_expanded_500/generations.json'
    cohort = root/'experiments/final_cohort_2026_10_24'
    saved = root/'experiments/expanded_model_selection_2026_10_24'
    provenance = root/'docs/engineering/baseline_mean_v2.json'
    for path in [targets.parent, generations.parent, cohort, saved, provenance.parent]:
        path.mkdir(parents=True, exist_ok=True)
    ids = ['p' + str(i) for i in range(486)]
    y = np.full(486, .1)
    rows = [dict(prompt_id='p'+str(i), dataset_index=i, target_eligible=i < 486,
                 n_generations=10, n_reliable=9, n_unreliable=1, n_inadequate=0,
                 n_manual_review=0, n_failed=0, n_missing=0, empirical_hallucination_risk=.1) for i in range(500)]
    targets.write_text(json.dumps(dict(evaluator_model='gemini-3.5-flash-lite', prompt_version='CP4.7.0', targets=rows)))
    generations.write_text('{}')
    def update_provenance():
        provenance.write_text(json.dumps({'source_sha256': {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in [targets,generations]}}))
    update_provenance()
    np.save(cohort/'pids.npy', ids)
    np.save(cohort/'y.npy', y)
    folds = []
    for i, idx in enumerate(np.array_split(np.arange(486), 5)):
        test = [ids[j] for j in idx]
        folds.append(dict(fold=i, train_pids=[p for p in ids if p not in test], test_pids=test))
    (saved/'fold_assignments.json').write_text(json.dumps({'fold_assignments':folds}))
    results = [dict(model='dummy_'+name, config=family, oof_pids=ids[::-1], oof_true=y.tolist(),
                    oof_predictions=y.tolist()) for name in ['mean','median'] for family in ['prompt','hidden','fusion']]
    (saved/'nested_cv_results.json').write_text(json.dumps({'results_all':results}))
    return targets, rows, update_provenance


def test_strict_snapshot_alignment_and_provenance(tmp_path):
    import json
    targets, rows, update = make_sources(tmp_path)
    loaded = module.load_sources(tmp_path)
    assert len(loaded[0]) == 486 and len(loaded[3]) == 5
    rows[0]['n_failed'] = 1
    targets.write_text(json.dumps(dict(evaluator_model='gemini-3.5-flash-lite', prompt_version='CP4.7.0', targets=rows)))
    with pytest.raises(ValueError, match='provenance'):
        module.load_sources(tmp_path)
    update()
    with pytest.raises(ValueError, match='strict target counts'):
        module.load_sources(tmp_path)


def test_policy_has_no_outcome_inputs_and_does_not_mutate_scores():
    scores = np.array([.1, .9])
    original = scores.copy()
    rule = module.ThresholdRule(.2)
    assert rule.route(scores).tolist() == rule.route(scores).tolist()
    assert np.array_equal(scores, original)
    with pytest.raises(TypeError):
        rule.route(scores, targets=np.array([1., 0.]))


def test_noncanonical_outer_fold_count_is_rejected(tmp_path):
    import json
    make_sources(tmp_path)
    path = tmp_path/'experiments/expanded_model_selection_2026_10_24/fold_assignments.json'
    d = json.loads(path.read_text())
    d['fold_assignments'] = d['fold_assignments'][:4]
    path.write_text(json.dumps(d))
    with pytest.raises(ValueError, match='original five'):
        module.load_sources(tmp_path)
