"""Export constants from the final strict snapshot without changing source data.

Run from repo root: python -m docs.engineering.export_baselines
Existing outputs are never overwritten. No evaluation or generation is performed.
"""
import hashlib
import json
from pathlib import Path

from src.predictor.inference import Artifact


TARGETS = 'data/production/cp4_7_0_gemini_3_5_flash_lite/targets_batch_05.json'
GENERATIONS = 'data/production/squad_v2_expanded_500/generations.json'
# This exporter intentionally supports only the audited frozen snapshot. A new
# cohort requires an explicit provenance review, not merely matching row counts.
AUDITED_HASHES = {
    TARGETS: '9f4a288540d41b31140248a690ad2759cff2d4cc66145fd0643ed4c28cf90914',
    GENERATIONS: '1a7babd61f817642d718c96e98b14d40196c0c60e99fbfc5dea463ccea16f3b8',
}


def validate_sources(target_bytes, generation_bytes):
    hashes = {TARGETS: hashlib.sha256(target_bytes).hexdigest(),
              GENERATIONS: hashlib.sha256(generation_bytes).hexdigest()}
    if hashes != AUDITED_HASHES:
        raise ValueError('Snapshot does not match the audited source provenance')
    return hashes


def main():
    source = Path(TARGETS)
    payload = source.read_bytes()
    provenance = validate_sources(payload, Path(GENERATIONS).read_bytes())
    rows = json.loads(payload)['targets']
    if len(rows) != 500 or len({r['prompt_id'] for r in rows}) != 500:
        raise ValueError('Unexpected final cohort')
    eligible = [r for r in rows if r['target_eligible'] is True]
    if len(eligible) != 486:
        raise ValueError('Unexpected strict eligible count')
    for row in eligible:
        fields = ['n_generations', 'n_reliable', 'n_unreliable', 'n_inadequate',
                  'n_manual_review', 'n_failed', 'n_missing']
        if any(type(row[k]) is not int or row[k] < 0 for k in fields):
            raise ValueError('Counts must be nonnegative integers')
        if any(row[k] != 0 for k in fields[3:]):
            raise ValueError('Excluded outcomes in eligible target')
        if row['n_generations'] != 10 or row['n_reliable'] + row['n_unreliable'] != 10:
            raise ValueError('Strict target eligibility mismatch')
        if row['empirical_hallucination_risk'] != row['n_unreliable'] / 10:
            raise ValueError('Risk/count mismatch')
    for statistic in ('mean', 'median'):
        artifact = Artifact.fit([r['empirical_hallucination_risk'] for r in eligible],
                                [r['prompt_id'] for r in eligible], statistic,
                                provenance)
        artifact.save(Path('docs/engineering') / ('baseline_' + statistic + '_v2.json'))


if __name__ == '__main__':
    main()
