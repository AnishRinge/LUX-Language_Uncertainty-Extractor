import unittest
import tempfile
import json
from pathlib import Path
from src.dataset.schema import CanonicalPromptScenario, SourceProvenance, PromptContent, ReferenceContent, ScenarioMetadata
from src.dataset.audit import audit_canonical_scenarios
from src.utils.hashing import compute_sha256

class TestAudit(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        
        # Create a mock raw file
        self.mock_raw_path = Path(self.tmp_dir.name) / "raw.json"
        self.mock_raw_path.write_text("mock raw content", encoding="utf-8")
        self.raw_sha = compute_sha256(self.mock_raw_path)

        # Create mock canonical scenarios
        self.scenarios = [
            CanonicalPromptScenario(
                sample_id="squad_v2_1",
                source=SourceProvenance(dataset="squad_v2", split="train", original_id="1", article_id="0", paragraph_id="0", article_title="Art1"),
                prompt=PromptContent(question="What is X?", context="Context about X."),
                reference=ReferenceContent(answerability="ANSWERABLE", answers=[{"text": "X"}]),
                metadata=ScenarioMetadata(phenomenon="grounded_contextual_reliability", expected_behavior="ANSWER")
            ),
            CanonicalPromptScenario(
                sample_id="squad_v2_2",
                source=SourceProvenance(dataset="squad_v2", split="train", original_id="2", article_id="0", paragraph_id="0", article_title="Art1"),
                prompt=PromptContent(question="What is Y?", context="Context about Y."),
                reference=ReferenceContent(answerability="UNANSWERABLE_FROM_CONTEXT", answers=[]),
                metadata=ScenarioMetadata(phenomenon="grounded_contextual_reliability", expected_behavior="ABSTAIN")
            )
        ]
        
        self.mock_canonical_path = Path(self.tmp_dir.name) / "canonical.json"
        with open(self.mock_canonical_path, "w", encoding="utf-8") as f:
            json.dump([s.to_dict() for s in self.scenarios], f)

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_audit_run_pass(self):
        report = audit_canonical_scenarios(self.mock_canonical_path, self.mock_raw_path, expected_hash=self.raw_sha)
        self.assertEqual(report["raw_artifact_integrity_pass"], True)
        self.assertEqual(report["record_counts"]["canonical_total"], 2)
        self.assertEqual(report["record_counts"]["canonical_answerable"], 1)
        self.assertEqual(report["record_counts"]["canonical_unanswerable"], 1)
        self.assertEqual(report["duplicate_statistics"]["duplicate_sample_id_count"], 0)
        self.assertEqual(report["forbidden_field_checks"]["forbidden_field_found"], False)

    def test_audit_forbidden_field_detection(self):
        # Inject forbidden field into scenario dict
        bad_scenario_dict = self.scenarios[0].to_dict()
        bad_scenario_dict["qwen_answer"] = "unauthorized answer"
        
        bad_canonical_path = Path(self.tmp_dir.name) / "bad_canonical.json"
        with open(bad_canonical_path, "w", encoding="utf-8") as f:
            json.dump([bad_scenario_dict, self.scenarios[1].to_dict()], f)
            
        report = audit_canonical_scenarios(bad_canonical_path, self.mock_raw_path, expected_hash=self.raw_sha)
        self.assertEqual(report["forbidden_field_checks"]["forbidden_field_found"], True)
        self.assertEqual(report["overall_status"], "FAIL")

    def test_audit_duplicate_sample_ids(self):
        # Create duplicate sample_ids
        scenarios_dup = [self.scenarios[0].to_dict(), self.scenarios[0].to_dict()]
        dup_canonical_path = Path(self.tmp_dir.name) / "dup_canonical.json"
        with open(dup_canonical_path, "w", encoding="utf-8") as f:
            json.dump(scenarios_dup, f)
            
        report = audit_canonical_scenarios(dup_canonical_path, self.mock_raw_path, expected_hash=self.raw_sha)
        self.assertGreater(report["duplicate_statistics"]["duplicate_sample_id_count"], 0)
        self.assertEqual(report["overall_status"], "FAIL")

if __name__ == "__main__":
    unittest.main()
