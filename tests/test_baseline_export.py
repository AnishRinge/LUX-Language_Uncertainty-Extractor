import pytest
from docs.engineering.export_baselines import validate_sources


def test_export_rejects_replaced_snapshot_or_generation_provenance():
    with pytest.raises(ValueError, match='audited source'):
        validate_sources(b'{"targets": []}', b'{"model": "different"}')
