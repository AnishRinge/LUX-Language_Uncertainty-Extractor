"""Verify this analysis's historical inputs and file inventory without writing them."""
import hashlib
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


if __name__ == '__main__':
    before = json.loads((HERE/'preservation_before.json').read_text(encoding='utf-8'))
    after = {str(f.relative_to(ROOT)): [f.stat().st_size, f.stat().st_mtime_ns]
             for root in ['data','experiments'] for f in (ROOT/root).rglob('*')
             if f.is_file() and HERE not in f.parents}
    # Inventory was recorded by Windows Path.__str__, using backslashes.
    changes = [name for name, value in before.items() if after.get(name) != value]
    added = sorted(set(after)-set(before))
    d = json.loads((HERE/'results.json').read_text(encoding='utf-8'))
    verified = {name: hashlib.sha256((ROOT/source['path']).read_bytes()).hexdigest() == source['sha256']
                for name,source in d['sources'].items()}
    code_match = hashlib.sha256((HERE/'analysis.py').read_bytes()).hexdigest() == d['analysis_sha256']
    contract_match = hashlib.sha256((HERE/'plan.md').read_bytes()).hexdigest() == d['contract_sha256']
    initial = json.loads((HERE/'results_initial.json').read_text(encoding='utf-8'))
    numerical_match = all(initial[k] == d[k] for k in ['policies','references','excluded'])
    diff = subprocess.run(['git','diff','--check'], cwd=ROOT, capture_output=True, text=True)
    report = dict(before_count=len(before), after_count=len(after), changed_or_removed=changes,
                  added_outside_new_directory=added, source_hashes_match=verified,
                  analysis_hash_matches=code_match, declared_contract_hash_matches=contract_match,
                  initial_final_routing_results_identical=numerical_match,
                  tracked_diff_check_returncode=diff.returncode,
                  scope='Metadata inventory for all historical data/experiments; SHA256 for all seven analysis input files. Not a hash of every historical byte.')
    report['passed'] = not changes and not added and all(verified.values()) and code_match and contract_match and numerical_match and diff.returncode == 0
    with (HERE/'preservation_check.json').open('x', encoding='utf-8') as f:
        json.dump(report,f,indent=2)
    print(json.dumps(report,indent=2))
    if not report['passed']:
        raise SystemExit('Preservation or reproducibility check failed')
