"""Read-only routing sensitivity analysis; never fits or selects a model/threshold.

From repo root: python experiments/routing_policy_evaluation_2026_10_27/analysis.py
Creates results.json exclusively. See plan.md for the declared evaluation contract.
"""
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
THRESHOLDS = (0., .05, .1, .2, .3, .5, .75, 1.)
BUDGETS = (0., .1, .25, .5, .75, 1.)
SALT = 'lux-routing-20261027'


def number(value, upper=None):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or (upper is not None and value > upper):
        raise ValueError('Expected a finite nonnegative number within declared bounds')


@dataclass(frozen=True)
class ThresholdRule:
    threshold: float
    high_action: str = 'verify'
    low_action: str = 'allow'

    def __post_init__(self):
        number(self.threshold, 1)
        if self.low_action != 'allow' or self.high_action not in ('verify', 'escalate'):
            raise ValueError('Unsupported routing actions')

    def route(self, scores):
        values = np.asarray(scores, dtype=float)
        if values.ndim != 1 or not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
            raise ValueError('Invalid risk scores; no clipping or zero substitution')
        return np.where(values >= self.threshold, self.high_action, self.low_action)


@dataclass(frozen=True)
class Costs:
    unreliable: float
    verify: float
    escalate: float

    def __post_init__(self):
        for value in (self.unreliable, self.verify, self.escalate):
            number(value)


def expected_cost(outcomes, actions, costs, residual_risk_fraction):
    """Offline scenario calculation; outcomes never enter the routing rule."""
    number(residual_risk_fraction, 1)
    y = np.asarray(outcomes, dtype=float)
    a = np.asarray(actions)
    if y.ndim != 1 or not len(y) or a.shape != y.shape or not set(a).issubset({'allow', 'verify', 'escalate'}):
        raise ValueError('Invalid outcome/action alignment')
    if not np.isfinite(y).all() or ((y < 0) | (y > 1)).any():
        raise ValueError('Invalid outcomes')
    remaining = y * np.where(a == 'allow', 1., residual_risk_fraction)
    return float(np.mean(remaining * costs.unreliable + (a == 'verify') * costs.verify + (a == 'escalate') * costs.escalate))


def require_selection_evidence(validation_ids, test_ids, training_ids_by_validation_id):
    """Guard for future selection: ordinary other-fold OOF scores fail this check.

    Evidence must enumerate all label-bearing fit/preprocessing/selection inputs,
    not merely IDs used in the final estimator fit. The current artifacts lack it.
    This is a lineage guard, not an optimizer; no selection is performed here.
    """
    if training_ids_by_validation_id is None or set(training_ids_by_validation_id) != set(validation_ids):
        raise ValueError('Missing score-producing pipeline lineage')
    if set(validation_ids) & set(test_ids) or len(set(validation_ids)) != len(validation_ids):
        raise ValueError('Selection/evaluation contamination')
    for pid in validation_ids:
        if (set(training_ids_by_validation_id[pid]) & set(test_ids)) or pid in training_ids_by_validation_id[pid]:
            raise ValueError('Score lineage contamination')


def align(row, ids, y):
    pids = row['oof_pids']
    if len(pids) != len(ids) or len(set(pids)) != len(pids) or set(pids) != set(ids):
        raise ValueError('OOF ID mismatch or duplicate')
    targets, predictions = np.asarray(row['oof_true'], dtype=float), np.asarray(row['oof_predictions'], dtype=float)
    if targets.shape != y.shape or predictions.shape != y.shape:
        raise ValueError('OOF shape mismatch')
    positions = {pid: i for i, pid in enumerate(ids)}
    order = np.array([positions[pid] for pid in pids])
    if not np.array_equal(targets, y[order]):
        raise ValueError('OOF target mismatch')
    result = np.empty_like(y)
    result[order] = predictions
    return result


def validate_folds(folds, ids):
    positions = {pid: i for i, pid in enumerate(ids)}
    if len(positions) != len(ids):
        raise ValueError('Duplicate cohort IDs')
    tested, fold_ids, result = [], [], []
    for fold in folds:
        tr, te = fold['train_pids'], fold['test_pids']
        if not tr or not te or len(set(tr)) != len(tr) or len(set(te)) != len(te) or set(tr) & set(te) or set(tr) | set(te) != set(ids):
            raise ValueError('Fold contamination or incomplete coverage')
        tested.extend(te)
        fold_ids.append(fold['fold'])
        result.append([positions[pid] for pid in te])
    if len(set(fold_ids)) != len(fold_ids) or len(tested) != len(ids) or set(tested) != set(ids):
        raise ValueError('Outer test coverage must be exactly once')
    return result


def metrics(y, escalated):
    y, escalated = np.asarray(y, dtype=float), np.asarray(escalated)
    if y.ndim != 1 or not len(y) or escalated.shape != y.shape or escalated.dtype != bool:
        raise ValueError('Invalid metric alignment/mask')
    if not np.isfinite(y).all() or ((y < 0) | (y > 1)).any():
        raise ValueError('Invalid risk targets')
    normal = ~escalated
    risk = float(y[normal].mean()) if normal.any() else None
    normal_mass, escalated_mass = float(y[normal].sum()), float(y[escalated].sum())
    return dict(n=len(y), n_normal=int(normal.sum()), n_escalated=int(escalated.sum()),
                normal_empirical_risk=risk, false_reassurance_generation_proxy=risk,
                normal_any_unreliable_prompt_fraction=float((y[normal] > 0).mean()) if normal.any() else None,
                escalation_fraction=float(escalated.mean()),
                captured_unreliable_fraction=escalated_mass/float(y.sum()) if y.sum() else None,
                hypothetical_remaining_per_original_prompt={str(q): (normal_mass + q * escalated_mass)/len(y) for q in (0., .5, 1.)})


def hash_mask(ids, fold_indices, counts):
    if len(counts) != len(fold_indices):
        raise ValueError('Budget/fold mismatch')
    selected = np.zeros(len(ids), dtype=bool)
    for idx, count in zip(fold_indices, counts):
        if type(count) is not int or not 0 <= count <= len(idx):
            raise ValueError('Invalid hash routing budget')
        order = sorted(idx, key=lambda i: (hashlib.sha256((SALT + ':' + ids[i]).encode()).hexdigest(), ids[i]))
        selected[order[:count]] = True
    return selected


def bootstrap_indices(fold_indices, replicates, seed):
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.choice(idx, size=(replicates, len(idx)), replace=True) for idx in fold_indices], axis=1)


def _sample_metrics(y, mask, indices):
    outcomes, selected = y[indices], mask[indices]
    normal = ~selected
    counts = normal.sum(axis=1)
    mass = (outcomes * normal).sum(axis=1)
    total = outcomes.sum(axis=1)
    def divide(a, b):
        return np.divide(a, b, out=np.full(a.shape, np.nan), where=b > 0)
    normal_risk = divide(mass, counts)
    return dict(normal_empirical_risk=normal_risk,
                false_reassurance_generation_proxy=normal_risk,
                normal_any_unreliable_prompt_fraction=divide(((outcomes > 0) * normal).sum(axis=1).astype(float), counts),
                escalation_fraction=selected.mean(axis=1),
                captured_unreliable_fraction=divide((outcomes * selected).sum(axis=1), total),
                remaining_per_original_prompt_if_perfect=mass/outcomes.shape[1])


def paired_intervals(y, candidate, reference, indices):
    a, b = _sample_metrics(y, candidate, indices), _sample_metrics(y, reference, indices)
    single = np.arange(len(y))[None, :]
    point_a, point_b = _sample_metrics(y, candidate, single), _sample_metrics(y, reference, single)
    report = {}
    for key in a:
        delta = a[key] - b[key]
        valid = np.isfinite(delta)
        point = point_a[key][0] - point_b[key][0]
        report[key] = dict(difference=float(point) if np.isfinite(point) else None,
                           interval_95=np.quantile(delta[valid], [.025, .975]).tolist() if valid.any() else None,
                           defined_replicates=int(valid.sum()), total_replicates=len(delta))
    return report


def load_sources(root):
    paths = dict(targets='data/production/cp4_7_0_gemini_3_5_flash_lite/targets_batch_05.json',
                 generations='data/production/squad_v2_expanded_500/generations.json',
                 y='experiments/final_cohort_2026_10_24/y.npy',
                 ids='experiments/final_cohort_2026_10_24/pids.npy',
                 folds='experiments/expanded_model_selection_2026_10_24/fold_assignments.json',
                 oof='experiments/expanded_model_selection_2026_10_24/nested_cv_results.json',
                 provenance='docs/engineering/baseline_mean_v2.json')
    hashes = {name: dict(path=path, sha256=hashlib.sha256((root/path).read_bytes()).hexdigest()) for name, path in paths.items()}
    artifact = json.loads((root/paths['provenance']).read_bytes())
    for name in ('targets', 'generations'):
        if artifact['source_sha256'].get(paths[name]) != hashes[name]['sha256']:
            raise ValueError('Audited source provenance mismatch')
    snapshot = json.loads((root/paths['targets']).read_bytes())
    if snapshot['evaluator_model'] != 'gemini-3.5-flash-lite' or snapshot['prompt_version'] != 'CP4.7.0':
        raise ValueError('Unsupported target evaluation version')
    rows = snapshot['targets']
    if len(rows) != 500 or len({r['prompt_id'] for r in rows}) != 500 or len({r['dataset_index'] for r in rows}) != 500:
        raise ValueError('Duplicate or incomplete strict snapshot')
    eligible = {r['prompt_id']: r for r in rows if r['target_eligible'] is True}
    ids = np.load(root/paths['ids'], allow_pickle=False).astype(str).tolist()
    y = np.load(root/paths['y'], allow_pickle=False)
    if len(ids) != 486 or len(set(ids)) != 486 or set(ids) != set(eligible) or y.shape != (486,):
        raise ValueError('Cohort/strict target mismatch')
    for pid, value in zip(ids, y):
        row = eligible[pid]
        counts = ['n_generations', 'n_reliable', 'n_unreliable', 'n_inadequate', 'n_manual_review', 'n_failed', 'n_missing']
        if any(type(row[k]) is not int or row[k] < 0 for k in counts) or any(row[k] for k in counts[3:]):
            raise ValueError('Invalid strict target counts')
        if row['n_generations'] != 10 or row['n_reliable'] + row['n_unreliable'] != 10 or value != row['n_unreliable']/10 or value != row['empirical_hallucination_risk']:
            raise ValueError('Risk/count mismatch')
    folds = json.loads((root/paths['folds']).read_bytes())['fold_assignments']
    if len(folds) != 5:
        raise ValueError('Expected the original five outer folds')
    indices = validate_folds(folds, ids)
    saved = json.loads((root/paths['oof']).read_bytes())
    aligned = {}
    for row in saved['results_all']:
        key = row['model'] + '|' + row['config']
        if key in aligned:
            raise ValueError('Duplicate model/configuration')
        aligned[key] = align(row, ids, y)
    for name in ('mean', 'median'):
        base = aligned['dummy_' + name + '|prompt']
        for family in ('hidden', 'fusion'):
            if not np.array_equal(base, aligned['dummy_' + name + '|' + family]):
                raise ValueError('Baseline copies disagree')
        for fold, idx in zip(folds, indices):
            train = [ids.index(pid) for pid in fold['train_pids']]
            expected = float(np.mean(y[train]) if name == 'mean' else np.median(y[train]))
            if not np.allclose(base[idx], expected, rtol=0, atol=1e-12):
                raise ValueError('Baseline uses non-training-fold targets')
    return ids, y, folds, indices, aligned, hashes, {k:v for k,v in saved.items() if k != 'results_all'}


def analyze(root):
    ids, y, folds, idx, scores, sources, metadata = load_sources(root)
    boot = bootstrap_indices(idx, 2000, 20261027)
    normal = np.zeros(len(y), dtype=bool)
    output = dict(schema_version=1, status='DESCRIPTIVE_ONLY_NO_SELECTED_POLICY', sources=sources,
                  seed=20261027, bootstrap_replicates=2000, thresholds=list(THRESHOLDS),
                  n=len(y), n_zero=int((y == 0).sum()), mean_risk=float(y.mean()),
                  unreliable_evaluator_outcomes=int(round(y.sum()*10)), oof_metadata=metadata,
                  selection=dict(performed=False, missing='Outer-model-specific inner validation scores and full fit/selection lineage'),
                  excluded={}, evaluated_configurations=[], policies={}, references={})
    output['method'] = dict(
        threshold_selection='None: declared grid applied in full; no winning threshold/model is selected',
        split='Original five outer test folds, exact-once 486-prompt coverage; ID-aligned for all policies',
        uncertainty='Paired prompt bootstrap stratified by fixed outer test fold; candidate minus reference',
        uncertainty_limits='Conditional on fixed masks/scores, not retraining variance; shared contexts and multiplicity unaccounted for',
        risk_semantics='UNRELIABLE/10 evaluator judgments per strictly eligible prompt; not next-answer certainty',
        verification_assumptions='Residual risk multipliers 0, .5, 1 on escalated prompts are hypothetical, not measured',
        false_reassurance='Mean empirical risk among normal-path prompts; also report fraction of such prompts with any unreliable attempt',
        remaining_risk_denominator='All original prompts; assumes verification returns answers with q times original empirical risk. For abstention use coverage and normal-path risk instead.',
        hash_reference='SHA256 fixed salt/ID ordering within each outer test fold; exact matched counts; no outcomes used',
        operational_impact='Not established; retrospective routing sensitivity only')
    output['analysis_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    output['contract_sha256'] = hashlib.sha256((HERE/'plan.md').read_bytes()).hexdigest()
    def record(mask):
        return dict(aggregate=metrics(y, mask), folds=[dict(fold=f['fold'], **metrics(y[j], mask[j])) for f,j in zip(folds,idx)],
                    escalated_prompt_ids=[pid for pid, selected in zip(ids,mask) if selected])
    output['references']['normal_generation'] = record(normal)
    for budget in BUDGETS:
        output['references']['hash_budget_' + str(budget)] = record(hash_mask(ids, idx, [int(budget*len(j)) for j in idx]))
    for key, values in scores.items():
        if key.startswith('dummy_') and not key.endswith('|prompt'):
            continue
        invalid = ~np.isfinite(values) | (values < 0) | (values > 1)
        if invalid.any():
            output['excluded'][key] = dict(reason='Risk-bound contract violated; configuration excluded, no prompt rows dropped or scores clipped',
                                          invalid_predictions=int(invalid.sum()))
            continue
        output['evaluated_configurations'].append(key)
        for threshold in THRESHOLDS:
            mask = ThresholdRule(threshold).route(values) != 'allow'
            matched = hash_mask(ids, idx, [int(mask[j].sum()) for j in idx])
            r = record(mask)
            r.update(predictor=key, threshold=threshold, action='verify',
                     evidence='verified train-fold baseline' if key.startswith('dummy_') else 'saved OOF; inner-fit provenance unverified',
                     matched_hash=record(matched),
                     paired_candidate_minus_normal=paired_intervals(y, mask, normal, boot),
                     paired_candidate_minus_matched_hash=paired_intervals(y, mask, matched, boot))
            output['policies'][key + '@' + str(threshold)] = r
    return output


if __name__ == '__main__':
    output = HERE/'results.json'
    if output.exists():
        raise FileExistsError('Never overwrite completed routing results; use a new experiment directory')
    results = analyze(ROOT)
    with output.open('x', encoding='utf-8') as f:
        json.dump(results, f, indent=2, allow_nan=False)
    print('Saved', len(results['policies']), 'descriptive policies;', len(results['excluded']), 'excluded configurations')
